/**
 * @file        xhttp.cpp
 * @brief       Online client API surface: NetDll_XHttp* (Xbox 360 HTTP client,
 *              WinHttp-alike) and the XampXAuth* authentication entry points.
 *
 * Implements the request lifecycle (Startup → Open → Connect → OpenRequest →
 * SendRequest → ReceiveResponse → Read/QueryHeaders → CloseHandle) with opaque
 * handles so titles progress past HTTP init instead of spinning on
 * XHttpStartup.
 *
 * Transport note: requests go out over WinHttp to whatever host the title was
 * configured with, so pointing it at a machine running tools/mktserver is all
 * it takes to bring the marketplace up. A host that does not answer falls back
 * to the in-process responder, which serves closet items and otherwise reports
 * 404, so titles take their offline path instead of hanging.
 *
 * @copyright   Copyright (c) 2026 Tom Clay <tomc@tctechstuff.com>
 * @license     BSD 3-Clause License, see LICENSE in the project root.
 */

#pragma GCC diagnostic ignored "-Wunused-parameter"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <rex/hook.h>
#include <rex/kernel/xnet.h>
#include <rex/logging.h>
#include <rex/runtime.h>
#include <rex/system/kernel_state.h>
#include <rex/string.h>
#include <rex/system/xthread.h>
#include <rex/system/xtypes.h>
#include <rex/types.h>

#include "avatars/closet.h"
#include "marketplace.h"

// game_data_root is defined by the runtime (runtime.cpp) at global scope.
REXCVAR_DECLARE(std::string, game_data_root);
// The store's server, defined next to the hive values in xam_user.cpp.
REXCVAR_DECLARE(std::string, avatar_marketplace_server);
REXCVAR_DECLARE(std::string, avatar_marketplace_key);

namespace rex {
namespace kernel {
namespace xam {

// Opaque XHTTP handles (HINTERNET-alike). Request handles carry state in
// g_requests so requests can be answered in-process: a title-storage GET is
// served here rather than over the network.
static std::atomic<uint32_t> g_xhttp_next_handle{0x58A00001u};

static inline uint32_t alloc_xhttp_handle() {
  return g_xhttp_next_handle.fetch_add(4, std::memory_order_relaxed);
}

// WinHttp/XHttp query info levels (subset).
static constexpr uint32_t kQueryStatusCode = 19;
static constexpr uint32_t kQueryCustom = 65;            // WINHTTP_QUERY_CUSTOM: header named by `name`
static constexpr uint32_t kQueryByNameXhttp = 0xFFFF;   // the XHTTP spelling of the same thing
static constexpr uint32_t kQueryRequestError = 0x29;    // final request error, 0 = completed cleanly
static constexpr uint32_t kQueryFlagNumber = 0x20000000u;
static constexpr uint32_t kQueryLevelMask = 0x1FFFFFFFu;

// Win32 errors the title reads back through RtlGetLastError after a FALSE.
static constexpr uint32_t kErrorInsufficientBuffer = 122;
static constexpr uint32_t kErrorHeaderNotFound = 12150;  // ERROR_WINHTTP_HEADER_NOT_FOUND

// WINHTTP_CALLBACK_STATUS_* notification codes (desktop WinHttp values; XHTTP
// mirrors them). The title opened the session with WINHTTP_FLAG_ASYNC
// (0x10000000) and registered a status callback, so completions are delivered
// by invoking that callback from XHttpDoWork rather than synchronously.
static constexpr uint32_t kCbHeadersAvailable = 0x00020000u;
static constexpr uint32_t kCbDataAvailable = 0x00040000u;
static constexpr uint32_t kCbReadComplete = 0x00080000u;
static constexpr uint32_t kCbSendRequestComplete = 0x00400000u;

struct XHttpRequest {
  std::string verb;
  std::string path;
  std::string host;          // from the connection this request was opened on
  uint16_t port = 80;
  std::string upload_body;   // PUT/POST request body (captured from XHttpSendRequest)
  int status = 0;            // HTTP status the emulator produced
  std::string headers;       // response headers, CRLF-separated "Name: value" lines
  std::string body;          // response body
  size_t read_offset = 0;    // XHttpReadData cursor
  uint32_t callback = 0;     // guest WINHTTP_STATUS_CALLBACK address
  uint32_t context = 0;      // dwContext passed to the callback
  uint32_t scratch = 0;      // guest DWORD scratch for DATA_AVAILABLE counts
};

// A connection the title opened, remembered so a request knows where to go.
struct XHttpConnection {
  std::string host;
  uint16_t port = 80;
};

// A pending async completion to deliver to a request's status callback.
struct XHttpNotification {
  uint32_t request_handle = 0;
  uint32_t status = 0;     // WINHTTP_CALLBACK_STATUS_*
  uint32_t info_ptr = 0;   // guest lpvStatusInformation
  uint32_t info_len = 0;   // dwStatusInformationLength
  uint32_t info_value = 0; // for DATA_AVAILABLE: byte count written to scratch
};

static std::mutex g_xhttp_mu;
static std::unordered_map<uint32_t, XHttpRequest> g_requests;  // request handle -> state
static std::unordered_map<uint32_t, XHttpConnection> g_connections;  // connect handle -> target
static std::vector<XHttpNotification> g_notifications;          // pending async completions

// Real transport. The title's hosts are configuration, not fiction: point them
// at a machine running the marketplace server and the store talks to it for
// real, which is the same path a public host would take.
static std::wstring Widen(const std::string& s) {
  return std::wstring(s.begin(), s.end());
}

// One session and one connect handle per host, kept open so WinHttp reuses
// the TLS connection instead of paying a handshake per tile.
static std::mutex g_http_mutex;
static HINTERNET g_http_session = nullptr;
static std::unordered_map<std::string, HINTERNET> g_http_connects;

static HINTERNET HttpConnection(const std::string& host, uint16_t port) {
  std::lock_guard<std::mutex> lock(g_http_mutex);
  if (!g_http_session) {
    g_http_session = WinHttpOpen(L"ReXGlue", WINHTTP_ACCESS_TYPE_NO_PROXY, WINHTTP_NO_PROXY_NAME,
                                 WINHTTP_NO_PROXY_BYPASS, 0);
    if (!g_http_session) {
      return nullptr;
    }
    WinHttpSetTimeouts(g_http_session, 5000, 5000, 15000, 30000);
  }
  const std::string key = host + ":" + std::to_string(port);
  auto it = g_http_connects.find(key);
  if (it != g_http_connects.end()) {
    return it->second;
  }
  HINTERNET connect = WinHttpConnect(g_http_session, Widen(host).c_str(), port, 0);
  if (connect) {
    g_http_connects[key] = connect;
  }
  return connect;
}

static bool HttpFetch(const std::string& host, uint16_t port, bool secure,
                      const std::string& verb, const std::string& path, int* status_out,
                      std::string* headers_out, std::string* body_out) {
  bool ok = false;
  HINTERNET connect = HttpConnection(host, port);
  if (connect) {
    HINTERNET request =
        WinHttpOpenRequest(connect, Widen(verb).c_str(), Widen(path).c_str(), nullptr,
                           WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                           secure ? WINHTTP_FLAG_SECURE : 0);
    if (request) {
      // The shared key rides on every request. Only the configured server ever
      // gets this far, so it goes nowhere else.
      const std::string key = REXCVAR_GET(avatar_marketplace_key);
      const std::wstring extra =
          key.empty() ? std::wstring() : L"X-Marketplace-Key: " + Widen(key) + L"\r\n";
      if (WinHttpSendRequest(request, extra.empty() ? WINHTTP_NO_ADDITIONAL_HEADERS : extra.c_str(),
                             extra.empty() ? 0 : static_cast<DWORD>(-1), WINHTTP_NO_REQUEST_DATA, 0,
                             0, 0) &&
          WinHttpReceiveResponse(request, nullptr)) {
        DWORD code = 0, code_size = sizeof(code);
        WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                            WINHTTP_HEADER_NAME_BY_INDEX, &code, &code_size,
                            WINHTTP_NO_HEADER_INDEX);
        *status_out = static_cast<int>(code);
        // The title reads Content-Type / Content-Encoding back by name, so keep
        // the raw header block and answer those queries from it.
        headers_out->clear();
        DWORD raw_size = 0;
        WinHttpQueryHeaders(request, WINHTTP_QUERY_RAW_HEADERS_CRLF, WINHTTP_HEADER_NAME_BY_INDEX,
                            WINHTTP_NO_OUTPUT_BUFFER, &raw_size, WINHTTP_NO_HEADER_INDEX);
        if (raw_size) {
          std::wstring raw(raw_size / sizeof(wchar_t), L'\0');
          if (WinHttpQueryHeaders(request, WINHTTP_QUERY_RAW_HEADERS_CRLF,
                                  WINHTTP_HEADER_NAME_BY_INDEX, raw.data(), &raw_size,
                                  WINHTTP_NO_HEADER_INDEX)) {
            raw.resize(raw_size / sizeof(wchar_t));
            headers_out->assign(raw.begin(), raw.end());
          }
        }
        body_out->clear();
        for (;;) {
          DWORD available = 0;
          if (!WinHttpQueryDataAvailable(request, &available) || !available) {
            break;
          }
          const size_t offset = body_out->size();
          body_out->resize(offset + available);
          DWORD read = 0;
          if (!WinHttpReadData(request, body_out->data() + offset, available, &read)) {
            body_out->resize(offset);
            break;
          }
          body_out->resize(offset + read);
        }
        ok = true;
      }
      WinHttpCloseHandle(request);
    }
  }
  return ok;
}

// avatar_marketplace_server is "scheme://host[:port]". The base it yields is
// spelled http:// whatever the scheme, because the title's own URL crackers
// only know that form; TLS is decided by the port instead, so https means 443.
MarketplaceServer GetMarketplaceServer() {
  MarketplaceServer server;
  std::string url = REXCVAR_GET(avatar_marketplace_server);
  while (!url.empty() && (url.back() == '/' || url.back() == ' ')) url.pop_back();
  const size_t scheme_end = url.find("://");
  std::string scheme = scheme_end == std::string::npos ? "http" : url.substr(0, scheme_end);
  for (auto& c : scheme) c = char(std::tolower(static_cast<unsigned char>(c)));
  const size_t host_start = scheme_end == std::string::npos ? 0 : scheme_end + 3;
  const size_t host_end = url.find('/', host_start);
  std::string hostport = url.substr(host_start, host_end == std::string::npos
                                                     ? std::string::npos
                                                     : host_end - host_start);
  server.secure = scheme == "https";
  server.port = server.secure ? 443 : 80;
  const size_t colon = hostport.rfind(':');
  if (colon != std::string::npos) {
    server.port = static_cast<uint16_t>(std::strtoul(hostport.c_str() + colon + 1, nullptr, 10));
    hostport.resize(colon);
  }
  for (auto& c : hostport) c = char(std::tolower(static_cast<unsigned char>(c)));
  server.host = hostport;
  server.valid = !server.host.empty() && server.port != 0;
  server.base = "http://" + server.host + ":" + std::to_string(server.port);
  return server;
}

static std::mutex g_games_filter_mutex;
static std::string g_games_filter;
static int g_filter_matches = -1;

void SetMarketplaceGamesFilter(const std::string& needle) {
  std::lock_guard<std::mutex> lock(g_games_filter_mutex);
  g_games_filter = needle;
  g_filter_matches = -1;
}

std::string MarketplaceGamesFilter() {
  std::lock_guard<std::mutex> lock(g_games_filter_mutex);
  return g_games_filter;
}

static std::string g_item_filter;
void SetMarketplaceItemFilter(const std::string& needle) {
  std::lock_guard<std::mutex> lock(g_games_filter_mutex);
  g_item_filter = needle;
  g_filter_matches = -1;
}
int MarketplaceFilterMatches() {
  std::lock_guard<std::mutex> lock(g_games_filter_mutex);
  return g_filter_matches;
}
std::string MarketplaceItemFilter() {
  std::lock_guard<std::mutex> lock(g_games_filter_mutex);
  return g_item_filter;
}

static std::string PercentEncode(const std::string& text) {
  static const char kHex[] = "0123456789ABCDEF";
  std::string out;
  for (unsigned char c : text) {
    if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
      out.push_back(char(c));
    } else {
      out.push_back('%');
      out.push_back(kHex[c >> 4]);
      out.push_back(kHex[c & 0xF]);
    }
  }
  return out;
}

// The Game Styles list is FindGames and item pages are FindGameOffers; while
// the matching filter is set the needle goes along as one more Names/Values
// pair.
static std::string WithGamesFilter(const std::string& path) {
  std::string needle;
  if (path.find("methodName=FindGames&") != std::string::npos) {
    needle = MarketplaceGamesFilter();
  } else if (path.find("methodName=FindGameOffers&") != std::string::npos) {
    needle = MarketplaceItemFilter();
  }
  if (needle.empty()) {
    return path;
  }
  return path + "&Names=NameFilter&Values=" + PercentEncode(needle);
}

bool MarketplaceGet(const std::string& path, std::string* body, std::string* headers) {
  const MarketplaceServer server = GetMarketplaceServer();
  if (!server.valid) {
    return false;
  }
  int status = 0;
  std::string raw_headers;
  const bool ok = HttpFetch(server.host, server.port, server.secure || server.port == 443, "GET",
                            path, &status, &raw_headers, body);
  if (headers) {
    *headers = raw_headers;
  }
  return ok && status == 200;
}

// Only the configured server gets a real connection. Catalog entries can still
// carry download.xbox.com art links and the title follows them; anything else
// is answered in-process so nothing leaves the machine unless the config says
// so.
static bool HostIsConfigured(const std::string& host) {
  std::string wanted = host;
  for (auto& c : wanted) c = char(std::tolower(static_cast<unsigned char>(c)));
  const MarketplaceServer server = GetMarketplaceServer();
  return server.valid && wanted == server.host;
}

// The Avatar Editor's item downloader builds "<root>/<titleid>/avataritems/
// <guid>.acp" (sub_920B5C20) and GETs it. The closet already holds those items
// under the same guid, so the bytes come straight off disk.
static bool ServeAvatarItem(XHttpRequest& r) {
  static constexpr std::string_view kMarker = "/avataritems/";
  const size_t pos = r.path.find(kMarker);
  if (pos == std::string::npos || r.verb != "GET") {
    return false;
  }
  std::string name = r.path.substr(pos + kMarker.size());
  name = name.substr(0, name.find('?'));
  static constexpr std::string_view kExt = ".acp";
  if (name.size() <= kExt.size() ||
      name.compare(name.size() - kExt.size(), kExt.size(), kExt) != 0) {
    return false;
  }
  name.resize(name.size() - kExt.size());

  avatars::AssetId id{};
  if (!avatars::ParseAssetId(name, &id)) {
    return false;
  }
  std::vector<uint8_t> bytes;
  if (!avatars::GetCloset().ReadItemBytes(id, bytes) || bytes.empty()) {
    // The catalog lists more items than the closet holds. A 404 is the honest
    // answer and the editor already handles a download that does not arrive.
    REXKRNL_INFO("[avatar-store] no local item for {}", name);
    return false;
  }
  r.body.assign(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  r.status = 200;
  return true;
}

// Fallback for when the configured host did not answer. Anything with no local
// answer reports 404: title-storage clients treat a missing blob as "use
// defaults" and proceed, rather than hanging on a connection error.
static void ServeRequest(XHttpRequest& r) {
  r.read_offset = 0;
  r.body.clear();
  r.headers.clear();
  if (ServeAvatarItem(r)) {
    // Same header surface the real transport would carry, so the title's
    // by-name lookups behave identically offline.
    r.headers = "Content-Type: application/octet-stream\r\nContent-Length: " +
                std::to_string(r.body.size()) + "\r\n";
    REXKRNL_INFO("[xhttp] {} {} -> 200 ({} bytes)", r.verb, r.path, r.body.size());
    return;
  }
  r.status = 404;
  REXKRNL_INFO("[xhttp] {} {} -> 404", r.verb, r.path);
}

// Case-insensitive lookup of one header in a CRLF-separated block.
static bool FindHeader(const std::string& block, const std::string& name, std::string* value) {
  size_t pos = 0;
  while (pos < block.size()) {
    size_t end = block.find("\r\n", pos);
    if (end == std::string::npos) end = block.size();
    const std::string line = block.substr(pos, end - pos);
    const size_t colon = line.find(':');
    if (colon != std::string::npos && colon == name.size()) {
      bool same = true;
      for (size_t i = 0; i < colon && same; ++i) {
        same = std::tolower(static_cast<unsigned char>(line[i])) ==
               std::tolower(static_cast<unsigned char>(name[i]));
      }
      if (same) {
        size_t v = colon + 1;
        while (v < line.size() && (line[v] == ' ' || line[v] == '\t')) ++v;
        *value = line.substr(v);
        return true;
      }
    }
    pos = end + 2;
  }
  return false;
}

// XHttpStartup(caller, reserved, reserved_ptr) -> BOOL.
u32 NetDll_XHttpStartup_entry(u32 caller, u32 reserved, u32 reserved_ptr) {
  return 1;
}

// XHttpShutdown(caller).
void NetDll_XHttpShutdown_entry(u32 caller) {}

// XHttpOpen(caller, agent, access_type, proxy, proxy_bypass, flags) -> hSession.
u32 NetDll_XHttpOpen_entry(u32 caller, mapped_string agent, u32 access_type,
                           mapped_string proxy, mapped_string proxy_bypass, u32 flags) {
  REXKRNL_DEBUG("XHttpOpen(agent='{}', access_type={}, flags={:#x})",
                agent ? static_cast<const char*>(agent) : "", access_type, flags);
  return alloc_xhttp_handle();
}

// XHttpConnect(caller, hSession, host, port, flags) -> hConnect.
u32 NetDll_XHttpConnect_entry(u32 caller, u32 session_handle, mapped_string host, u32 port,
                              u32 flags) {
  const uint32_t handle = alloc_xhttp_handle();
  XHttpConnection connection;
  connection.host = host ? static_cast<const char*>(host) : "";
  connection.port = port ? static_cast<uint16_t>(port) : uint16_t(80);
  REXKRNL_INFO("XHttpConnect(session={:#x}, host='{}', port={}) -> connect={:#x}", session_handle,
               connection.host, connection.port, handle);
  {
    std::lock_guard<std::mutex> lock(g_xhttp_mu);
    g_connections[handle] = std::move(connection);
  }
  return handle;
}

// XHttpOpenRequest(caller, hConnect, verb, path, version, referrer, accept_types, flags)
//   -> hRequest.
u32 NetDll_XHttpOpenRequest_entry(u32 caller, u32 connect_handle, mapped_string verb,
                                  mapped_string path, mapped_string version,
                                  mapped_string referrer, u32 accept_types, u32 flags) {
  const uint32_t handle = alloc_xhttp_handle();
  XHttpRequest req;
  req.verb = verb ? static_cast<const char*>(verb) : "GET";
  req.path = path ? static_cast<const char*>(path) : "";
  REXKRNL_DEBUG("XHttpOpenRequest(connect={:#x}) {} {} -> request={:#x}", connect_handle, req.verb,
                req.path, handle);
  {
    std::lock_guard<std::mutex> lock(g_xhttp_mu);
    auto connection = g_connections.find(connect_handle);
    if (connection != g_connections.end()) {
      req.host = connection->second.host;
      req.port = connection->second.port;
    }
    g_requests[handle] = std::move(req);
  }
  return handle;
}

// XHttpOpenRequestUsingMemory(caller, hConnect, verb, path, version, referrer,
//                             accept_types, flags) -> hRequest.
// The editor's net client opens every request through this variant and does not
// import XHttpOpenRequest at all. Same arguments; the "memory" is the caller's
// own request pool, which nothing here needs to honour. Left stubbed it handed
// back a null handle that the request thread then dereferenced.
u32 NetDll_XHttpOpenRequestUsingMemory_entry(u32 caller, u32 connect_handle, mapped_string verb,
                                             mapped_string path, mapped_string version,
                                             mapped_string referrer, u32 accept_types, u32 flags) {
  REXKRNL_INFO(
      "XHttpOpenRequestUsingMemory(connect={:#x}) verb='{}' path='{}' version='{}' "
      "referrer='{}' accept={:#x} flags={:#x}",
      connect_handle, verb ? static_cast<const char*>(verb) : "",
      path ? static_cast<const char*>(path) : "",
      version ? static_cast<const char*>(version) : "",
      referrer ? static_cast<const char*>(referrer) : "", accept_types, flags);
  return NetDll_XHttpOpenRequest_entry(caller, connect_handle, verb, path, version, referrer,
                                       accept_types, flags);
}

// XHttpSetStatusCallback(caller, handle, callback, flags, reserved) -> prev callback.
u32 NetDll_XHttpSetStatusCallback_entry(u32 caller, u32 handle, u32 callback_ptr, u32 flags,
                                        u32 reserved) {
  REXKRNL_INFO("XHttpSetStatusCallback(handle={:#x}, callback={:#x}, flags={:#x})", handle,
               callback_ptr, flags);
  std::lock_guard<std::mutex> lock(g_xhttp_mu);
  auto it = g_requests.find(handle);
  if (it != g_requests.end()) {
    it->second.callback = callback_ptr;
  }
  return 1;
}

// XHttpSendRequest(caller, hRequest, headers, header_len, optional, optional_len,
//                  total_len, context) -> BOOL. Produces the in-process response.
u32 NetDll_XHttpSendRequest_entry(u32 caller, u32 request_handle, mapped_string headers,
                                  u32 header_len, u32 optional_ptr, u32 optional_len,
                                  u32 total_len, u32 context) {
  REXKRNL_DEBUG("XHttpSendRequest(request={:#x}, header_len={}, context={:#x})", request_handle,
                header_len, context);
  std::string verb, path, host;
  uint16_t port = 80;
  {
    std::lock_guard<std::mutex> lock(g_xhttp_mu);
    auto it = g_requests.find(request_handle);
    if (it == g_requests.end()) {
      return 0;
    }
    auto& r = it->second;
    r.context = context;
    // Capture the request body (PUT/POST payload) from the optional buffer so
    // the emulated Title Storage can persist it.
    r.upload_body.clear();
    if (optional_ptr && optional_len) {
      const auto* p = REX_KERNEL_MEMORY()->TranslateVirtual<const char*>(optional_ptr);
      if (p) {
        r.upload_body.assign(p, optional_len);
      }
    }
    verb = r.verb;
    path = r.path;
    host = r.host;
    port = r.port;
  }

  // Anything not meant for the configured server is answered in-process.
  const bool remote = !host.empty() && HostIsConfigured(host);
  if (!host.empty() && !remote) {
    REXKRNL_INFO("[xhttp] {} http://{}:{}{} -> not a configured host, answered locally", verb,
                 host, port, path);
  }
  if (!remote) {
    std::lock_guard<std::mutex> lock(g_xhttp_mu);
    auto it = g_requests.find(request_handle);
    if (it == g_requests.end()) {
      return 0;
    }
    ServeRequest(it->second);
    g_notifications.push_back({request_handle, kCbSendRequestComplete, 0, 0, 0});
    return 1;
  }

  // The fetch runs on its own thread and completes through DoWork, so the
  // title keeps drawing its spinner and its parallel requests overlap.
  std::thread([request_handle, verb, path, host, port]() {
    int status = 0;
    std::string response_headers, body;
    const std::string request_path = WithGamesFilter(path);
    const bool fetched = HttpFetch(host, port, port == 443, verb, request_path, &status,
                                   &response_headers, &body);
    if (fetched) {
      REXKRNL_INFO("[xhttp] {} http://{}:{}{} -> {} ({} bytes)", verb, host, port, request_path,
                   status, body.size());
    }
    // A filtered list answers with the match count the search box shows.
    if (fetched && status == 200 && request_path.find("&Names=NameFilter&") != std::string::npos) {
      const size_t at = body.find("<totalItems>");
      if (at != std::string::npos) {
        std::lock_guard<std::mutex> lock(g_games_filter_mutex);
        g_filter_matches = std::atoi(body.c_str() + at + 12);
      }
    }
    // A catalog page lists its items as urn:uuid entries. Their art is fetched
    // before the page is handed over, the packages after, in the background.
    if (fetched && status == 200 && path.find("methodName=FindGameOffers") != std::string::npos) {
      std::vector<std::string> guids;
      static const char kUrn[] = "urn:uuid:";
      for (size_t pos = body.find(kUrn); pos != std::string::npos;
           pos = body.find(kUrn, pos + 1)) {
        const std::string guid = body.substr(pos + sizeof(kUrn) - 1, 36);
        if (guid.size() == 36 && guid.compare(0, 4, "0000") == 0) {
          guids.push_back(guid);
        }
      }
      MarketplacePrefetchIcons(guids, true);
      MarketplacePrefetchItems(guids);
    }
    std::lock_guard<std::mutex> lock(g_xhttp_mu);
    auto it = g_requests.find(request_handle);
    if (it == g_requests.end()) {
      return;  // closed while in flight
    }
    auto& r = it->second;
    if (fetched) {
      r.read_offset = 0;
      r.status = status;
      r.headers = std::move(response_headers);
      r.body = std::move(body);
    } else {
      ServeRequest(r);
    }
    g_notifications.push_back({request_handle, kCbSendRequestComplete, 0, 0, 0});
  }).detach();
  return 1;
}

// XHttpReceiveResponse(caller, hRequest, reserved) -> BOOL.
// Async: headers become available, and the title's callback then issues its
// read (it does not poll QueryDataAvailable).
u32 NetDll_XHttpReceiveResponse_entry(u32 caller, u32 request_handle, u32 reserved) {
  REXKRNL_INFO("XHttpReceiveResponse(request={:#x})", request_handle);
  std::lock_guard<std::mutex> lock(g_xhttp_mu);
  auto it = g_requests.find(request_handle);
  if (it == g_requests.end()) {
    return 0;
  }
  // The title's auth-manager callback is state-machine driven: HEADERS_AVAILABLE
  // sets the "headers ready" state and its loop then issues the read. It treats
  // DATA_AVAILABLE (0x40000) as an invalid state, so that must not be sent.
  g_notifications.push_back({request_handle, kCbHeadersAvailable, 0, 0, 0});
  return 1;
}

// XHttpReadData(caller, hRequest, buffer, bytes_to_read, bytes_read_ptr) -> BOOL.
// Async: copies up to bytes_to_read into the guest buffer now and reports the
// count via a READ_COMPLETE notification on the next DoWork (lpvStatusInformation
// = buffer, dwStatusInformationLength = bytes read; 0 bytes => end of data).
u32 NetDll_XHttpReadData_entry(u32 caller, u32 request_handle, u32 buffer_guest,
                               u32 bytes_to_read, u32 bytes_read_ptr) {
  REXKRNL_INFO("XHttpReadData(request={:#x}, buffer={:#x}, to_read={})", request_handle,
               buffer_guest, bytes_to_read);
  std::lock_guard<std::mutex> lock(g_xhttp_mu);
  auto it = g_requests.find(request_handle);
  if (it == g_requests.end()) {
    return 0;
  }
  auto& r = it->second;
  uint32_t n = 0;
  if (buffer_guest) {
    const size_t remaining = r.body.size() - r.read_offset;
    n = static_cast<uint32_t>(std::min<size_t>(bytes_to_read, remaining));
    if (n) {
      std::memcpy(REX_KERNEL_MEMORY()->TranslateVirtual<uint8_t*>(buffer_guest),
                  r.body.data() + r.read_offset, n);
      r.read_offset += n;
    }
  }
  if (bytes_read_ptr) {
    *REX_KERNEL_MEMORY()->TranslateVirtual<rex::be<uint32_t>*>(bytes_read_ptr) = n;
  }
  // Async: report the read result. A subsequent read returning 0 bytes signals
  // EOF (READ_COMPLETE with length 0). No DATA_AVAILABLE, see ReceiveResponse.
  REXKRNL_INFO("XHttpReadData(request={:#x}) -> {} bytes, offset {}/{}", request_handle, n,
               r.read_offset, r.body.size());
  g_notifications.push_back({request_handle, kCbReadComplete, buffer_guest, n, 0});
  return 1;
}

// XHttpQueryHeaders(caller, hRequest, info_level, name, buffer, buffer_len_ptr, index) -> BOOL.
u32 NetDll_XHttpQueryHeaders_entry(u32 caller, u32 request_handle, u32 info_level,
                                   mapped_string name, mapped_void buffer,
                                   mapped_u32 buffer_len_ptr, mapped_u32 index_ptr) {
  REXKRNL_INFO("XHttpQueryHeaders(request={:#x}, info_level={:#x}, name='{}')", request_handle,
               info_level, name ? static_cast<const char*>(name) : "");
  std::lock_guard<std::mutex> lock(g_xhttp_mu);
  auto it = g_requests.find(request_handle);
  if (it == g_requests.end()) {
    return 0;
  }
  const uint32_t level = info_level & kQueryLevelMask;
  const bool as_number = (info_level & kQueryFlagNumber) != 0;
  auto& req = it->second;

  // Header by name. The store's response stage (sub_9220B038) asks for
  // Content-Encoding this way and only tolerates two answers: the value
  // "gzip", or a FALSE whose last error is ERROR_WINHTTP_HEADER_NOT_FOUND,
  // which it reads as identity encoding. Anything else aborts the response
  // before it is parsed. The length it checks counts the terminator.
  const char* wanted = name ? static_cast<const char*>(name) : nullptr;
  if ((level == kQueryByNameXhttp || level == kQueryCustom) && wanted && *wanted) {
    std::string value;
    if (!FindHeader(req.headers, wanted, &value)) {
      REXKRNL_INFO("[xhttp] header '{}' absent", wanted);
      rex::system::XThread::SetLastError(kErrorHeaderNotFound);
      return 0;
    }
    const uint32_t needed = static_cast<uint32_t>(value.size() + 1);
    if (!buffer || !buffer_len_ptr || uint32_t(*buffer_len_ptr) < needed) {
      if (buffer_len_ptr) {
        *buffer_len_ptr = needed;
      }
      rex::system::XThread::SetLastError(kErrorInsufficientBuffer);
      return 0;
    }
    std::memcpy(static_cast<uint8_t*>(buffer), value.c_str(), needed);
    *buffer_len_ptr = needed;
    REXKRNL_INFO("[xhttp] header '{}' = '{}'", wanted, value);
    return 1;
  }

  // The title (ATGHTTP) queries the status code first (level 0xFFFE here, also
  // accept the desktop WinHttp value 19) then the content length (level 9).
  // Both are requested as numbers.
  const bool is_status = (level == 0xFFFE) || (level == kQueryStatusCode);

  if (as_number) {
    // The answer takes the caller's width. The store's client (sub_9228EF10)
    // fetches the content length into a 64-bit slot and then tests its low
    // dword; a 32-bit write lands in the high half on big-endian, the low half
    // reads as zero, and the pump closes the request without ever reading the
    // body. The title-storage client passes 4 and keeps getting 32 bits.
    // Level 0x29 is queried by the pump (sub_9228EF10) at end-of-read straight
    // into its result slot, and a successful answer is returned to the driver
    // verbatim as the completion code. The driver only understands its own
    // sentinels there (0x150070 done, 0x150071/2 pending): a content length
    // reads as an error and a zero re-arms the poll forever. Failing the query
    // makes the pump fall through to its status check and produce the done
    // sentinel itself.
    if (level == kQueryRequestError) {
      rex::system::XThread::SetLastError(kErrorHeaderNotFound);
      return 0;
    }
    const uint64_t value = is_status ? uint64_t(req.status) : uint64_t(req.body.size());
    const uint32_t buf_len = buffer_len_ptr ? uint32_t(*buffer_len_ptr) : 0;
    const uint32_t width = buf_len >= sizeof(uint64_t) ? 8u : buf_len >= sizeof(uint32_t) ? 4u : 0u;
    REXKRNL_INFO("  QueryHeaders num: level={:#x} is_status={} value={} buf_len={} -> {}", level,
                 is_status, value, buf_len, width ? "OK" : "FAIL");
    if (!buffer || !buffer_len_ptr || !width) {
      if (buffer_len_ptr) {
        *buffer_len_ptr = sizeof(uint32_t);
      }
      rex::system::XThread::SetLastError(kErrorInsufficientBuffer);
      return 0;
    }
    if (width == 8) {
      *reinterpret_cast<rex::be<uint64_t>*>(static_cast<uint8_t*>(buffer)) = value;
    } else {
      *reinterpret_cast<rex::be<uint32_t>*>(static_cast<uint8_t*>(buffer)) =
          static_cast<uint32_t>(value);
    }
    *buffer_len_ptr = width;
    return 1;
  }

  // String form (e.g. status text): return the numeric value as text.
  const std::string s =
      is_status ? std::to_string(req.status) : std::to_string(req.body.size());
  if (buffer && buffer_len_ptr && uint32_t(*buffer_len_ptr) >= s.size() + 1) {
    std::memcpy(static_cast<uint8_t*>(buffer), s.c_str(), s.size() + 1);
    *buffer_len_ptr = static_cast<uint32_t>(s.size());
    return 1;
  }
  return 0;
}

// XHttpDoWork(caller, handle, reserved) -> BOOL.
// Pumps the async state machine: delivers a queued completion to the request's
// status callback. The callback re-enters XHttp (ReceiveResponse/ReadData) and
// enqueues the next completion, so the lock is released around the callback to
// permit that re-entrancy.
u32 NetDll_XHttpDoWork_entry(u32 caller, u32 handle, u32 reserved) {
  REXKRNL_DEBUG("XHttpDoWork(handle={:#x})", handle);
  // Deliver one completion per pump (real async semantics): the title defers
  // its header processing, so draining the whole queue at once would push
  // DATA_AVAILABLE before it is ready to read.
  for (uint32_t guard = 0; guard < 1; ++guard) {
    XHttpNotification n;
    uint32_t callback = 0;
    uint32_t context = 0;
    {
      std::lock_guard<std::mutex> lock(g_xhttp_mu);
      if (g_notifications.empty()) {
        break;
      }
      n = g_notifications.front();
      g_notifications.erase(g_notifications.begin());
      auto it = g_requests.find(n.request_handle);
      if (it != g_requests.end()) {
        callback = it->second.callback;
        context = it->second.context;
        // DATA_AVAILABLE delivers a pointer to a DWORD byte-count. Stage it in
        // the request's guest scratch DWORD.
        if (n.status == kCbDataAvailable) {
          if (!it->second.scratch) {
            it->second.scratch = REX_KERNEL_MEMORY()->SystemHeapAlloc(sizeof(uint32_t));
          }
          *REX_KERNEL_MEMORY()->TranslateVirtual<rex::be<uint32_t>*>(it->second.scratch) =
              n.info_value;
          n.info_ptr = it->second.scratch;
          n.info_len = sizeof(uint32_t);
        }
      }
    }
    if (!callback) {
      continue;
    }
    PPCFunc* fn = rex::runtime::ResolveIndirectFunction(callback);
    if (!fn) {
      REXKRNL_WARN("XHttpDoWork: unresolved status callback {:#x}", callback);
      continue;
    }
    REXKRNL_INFO("XHttp callback(request={:#x}, status={:#x}, info={:#x}, len={})",
                 n.request_handle, n.status, n.info_ptr, n.info_len);
    // WINHTTP_STATUS_CALLBACK(hInternet, dwContext, dwInternetStatus,
    //                         lpvStatusInformation, dwStatusInformationLength)
    rex::ppc::GuestToHostFunction<void>(fn, n.request_handle, context, n.status, n.info_ptr,
                                        n.info_len);
  }
  // BOOL success, not a pending/delivered count: request drivers call DoWork
  // after every state step and treat 0 as a hard error, and an idle pump with
  // an empty notification queue is normal.
  return 1;
}

// XHttpCloseHandle(caller, handle) -> BOOL.
u32 NetDll_XHttpCloseHandle_entry(u32 caller, u32 handle) {
  REXKRNL_INFO("XHttpCloseHandle(handle={:#x})", handle);
  std::lock_guard<std::mutex> lock(g_xhttp_mu);
  g_requests.erase(handle);
  g_connections.erase(handle);
  return 1;
}

// XHttpQueryOption(caller, handle, option, buffer, buffer_len_ptr) -> BOOL.
// The store's driver reads two connection-health options as DWORDs and treats
// a failed query as a dead connection: option 22 in its reuse decision
// (sub_9220AF98, "may this connection be kept?") and option 23 in its
// post-connect check (sub_9220B948, "verified?", bit 0). Both tear the request
// down on failure, so answer them as healthy. Anything else still fails, and
// is logged so the next one the guest asks for is visible.
u32 NetDll_XHttpQueryOption_entry(u32 caller, u32 handle, u32 option, mapped_void buffer,
                                  mapped_u32 buffer_len_ptr) {
  const uint32_t buf_len = buffer_len_ptr ? uint32_t(*buffer_len_ptr) : 0;
  if ((option == 22 || option == 23) && buffer && buffer_len_ptr && buf_len >= sizeof(uint32_t)) {
    *reinterpret_cast<rex::be<uint32_t>*>(static_cast<uint8_t*>(buffer)) = 1u;
    *buffer_len_ptr = sizeof(uint32_t);
    REXKRNL_INFO("XHttpQueryOption(handle={:#x}, option={}) -> 1", handle, option);
    return 1;
  }
  REXKRNL_INFO("XHttpQueryOption(handle={:#x}, option={}, buf_len={}) unsupported", handle, option,
               buf_len);
  return 0;
}

// XHttpSetOption(caller, handle, option, buffer, buffer_len) -> BOOL.
u32 NetDll_XHttpSetOption_entry(u32 caller, u32 handle, u32 option, mapped_void buffer,
                                u32 buffer_len) {
  REXKRNL_DEBUG("XHttpSetOption(handle={:#x}, option={})", handle, option);
  return 1;
}

//=============================================================================
// XampXAuth*, Xbox authentication. The title spins up XAuth as part of its
// online/service login; report success so it proceeds (mirrors xenia-canary
// with network_mode=XBOXLIVE). Left stubbed, the title enters a startup-retry
// storm that exhausts the guest heap.
//=============================================================================

// XampXAuthStartup(settings) -> X_ERROR_SUCCESS. Single arg = XAUTH_SETTINGS*.
u32 XampXAuthStartup_entry(u32 settings_ptr) {
  return 0;  // X_ERROR_SUCCESS
}

// XampXAuthShutdown(unkn) -> void. Writes 1 to *unkn (0 would request a
// follow-up XampXAuthGetTitleBuffer call).
void XampXAuthShutdown_entry(mapped_u32 unkn_ptr) {
  if (unkn_ptr) {
    *unkn_ptr = 1;
  }
}

// XampXAuthGetTitleBuffer() -> 0 (a non-zero pointer here would crash the title).
u32 XampXAuthGetTitleBuffer_entry() {
  return 0;
}

// XampXAuthIsLocalSocketAllowed() -> TRUE (permit SO_GRANTINSECURE sockets).
u32 XampXAuthIsLocalSocketAllowed_entry() {
  return 1;
}

//=============================================================================
// XamGetServiceEndpoint(service_name, out_endpoint, out_len, overlapped)
//   The title asks the console to resolve a named live service to a URL. On
//   real hardware the Xbox live service directory returned the publisher's
//   server; here it resolves to a locally controlled host so requests land on
//   the in-process service emulator.
//=============================================================================

// Host the title is pointed at. Requests are intercepted in the XHttp layer, so
// this never has to be reachable on the network; the path carries the service.
static constexpr const char* kServiceHost = "http://service.rexglue/";

u32 XamGetServiceEndpoint_entry(mapped_string service_name, mapped_string out_endpoint,
                                u32 out_endpoint_len, u32 overlapped_ptr) {
  if (!service_name || !out_endpoint || !out_endpoint_len) {
    return X_ERROR_INVALID_PARAMETER;
  }

  std::string name(static_cast<const char*>(service_name));
  std::string endpoint = std::string(kServiceHost) + name;

  X_RESULT result;
  std::memset(static_cast<char*>(out_endpoint), 0, out_endpoint_len);
  if (out_endpoint_len < endpoint.size() + 1) {
    result = X_ERROR_INSUFFICIENT_BUFFER;
  } else {
    std::memcpy(static_cast<char*>(out_endpoint), endpoint.c_str(), endpoint.size() + 1);
    result = X_ERROR_SUCCESS;
  }

  if (overlapped_ptr) {
    rex::system::kernel_state()->CompleteOverlappedImmediate(overlapped_ptr, result);
    return X_ERROR_IO_PENDING;
  }
  return result;
}

// XamGetToken(user_index, url, url_size, token_out, overlapped) -> result.
// Returns a mock relying-party (XSTS) token; the title attaches it to the
// Authorization header of the title-storage request, and nothing validates it.
// Mirrors xenia-canary's mock token.
u32 XamGetToken_entry(u32 user_index, mapped_string url, u32 url_size, mapped_u32 token_out,
                      u32 overlapped_ptr) {
  if (user_index >= 4) {
    return X_ERROR_INVALID_PARAMETER;
  }
  auto* mem = REX_KERNEL_MEMORY();

  static const char kMockToken[] = "MOCK_XBOX_STS_TOKEN";
  const uint32_t token_len = static_cast<uint32_t>(sizeof(kMockToken) - 1);

  const uint32_t data_addr = mem->SystemHeapAlloc(token_len);
  std::memcpy(mem->TranslateVirtual<uint8_t*>(data_addr), kMockToken, token_len);

  const uint32_t token_addr = mem->SystemHeapAlloc(sizeof(rex::kernel::XAM_RELYING_PARTY_TOKEN));
  auto* token = mem->TranslateVirtual<rex::kernel::XAM_RELYING_PARTY_TOKEN*>(token_addr);
  token->reserved = 0;
  token->length = token_len;
  token->token_data_ptr = data_addr;

  if (token_out) {
    *token_out = token_addr;
  }

  if (overlapped_ptr) {
    rex::system::kernel_state()->CompleteOverlappedImmediate(overlapped_ptr, X_ERROR_SUCCESS);
    return X_ERROR_IO_PENDING;
  }
  return X_ERROR_SUCCESS;
}

// XHttpCrackUrl(caller, url, url_length, flags, components) -> BOOL.
// Parses scheme://host[:port][/path][?query] into the guest XHTTP_URL_COMPONENTS.
// Supports both modes the title may use per component: a caller-provided buffer
// (ptr set -> copy + NUL) or pointer-into-URL (ptr 0, length nonzero -> set ptr
// to point into the original URL). Enough for the live service endpoints handed
// out by XamGetServiceEndpoint.
u32 NetDll_XHttpCrackUrl_entry(u32 caller, u32 url_guest, u32 url_length, u32 flags,
                               u32 components_guest) {
  auto* mem = REX_KERNEL_MEMORY();
  if (!url_guest || !components_guest) {
    return 0;
  }
  auto* comp = mem->TranslateVirtual<rex::kernel::XHTTP_URL_COMPONENTS*>(components_guest);
  if (comp->struct_size != sizeof(rex::kernel::XHTTP_URL_COMPONENTS)) {
    return 0;
  }

  const char* url_host = mem->TranslateVirtual<const char*>(url_guest);
  std::string url = url_length ? std::string(url_host, url_length) : std::string(url_host);
  REXKRNL_DEBUG("XHttpCrackUrl('{}')", url);

  size_t scheme_off = 0, scheme_len = 0, host_off = 0, host_len = 0;
  size_t path_off = 0, path_len = 0, query_off = 0, query_len = 0;
  uint16_t port = 0;
  uint32_t scheme_enum = 0;

  size_t cursor = 0;
  const size_t scheme_sep = url.find("://");
  if (scheme_sep != std::string::npos) {
    scheme_off = 0;
    scheme_len = scheme_sep;
    cursor = scheme_sep + 3;
    const std::string scheme = url.substr(0, scheme_len);
    if (scheme == "https") {
      scheme_enum = static_cast<uint32_t>(rex::kernel::X_INTERNET_SCHEME::HTTPS);
      port = 443;
    } else {
      scheme_enum = static_cast<uint32_t>(rex::kernel::X_INTERNET_SCHEME::HTTP);
      port = 80;
    }
  }

  const size_t host_start = cursor;
  const size_t path_start = url.find('/', cursor);
  const size_t hostport_end = (path_start == std::string::npos) ? url.size() : path_start;
  const std::string hostport = url.substr(host_start, hostport_end - host_start);
  const size_t colon = hostport.find(':');
  if (colon != std::string::npos) {
    host_off = host_start;
    host_len = colon;
    port = static_cast<uint16_t>(std::strtoul(hostport.c_str() + colon + 1, nullptr, 10));
  } else {
    host_off = host_start;
    host_len = hostport.size();
  }

  if (path_start != std::string::npos) {
    const size_t q = url.find('?', path_start);
    if (q != std::string::npos) {
      path_off = path_start;
      path_len = q - path_start;
      query_off = q;
      query_len = url.size() - q;
    } else {
      path_off = path_start;
      path_len = url.size() - path_start;
    }
  }

  comp->scheme = scheme_enum;
  comp->port = port;

  auto emit = [&](rex::be<uint32_t>& ptr_field, rex::be<uint32_t>& len_field, size_t off,
                  size_t len) -> bool {
    const uint32_t cur_ptr = ptr_field;
    const uint32_t cur_len = len_field;
    if (cur_ptr) {
      if (cur_len < len + 1) {
        len_field = static_cast<uint32_t>(len + 1);
        return false;
      }
      char* dst = mem->TranslateVirtual<char*>(cur_ptr);
      std::memcpy(dst, url.data() + off, len);
      dst[len] = '\0';
      len_field = static_cast<uint32_t>(len);
    } else if (cur_len) {
      ptr_field = url_guest + static_cast<uint32_t>(off);
      len_field = static_cast<uint32_t>(len);
    }
    return true;
  };

  bool ok = true;
  if (scheme_len) ok &= emit(comp->scheme_ptr, comp->scheme_length, scheme_off, scheme_len);
  if (host_len) ok &= emit(comp->host_name_ptr, comp->host_name_length, host_off, host_len);
  if (path_len) ok &= emit(comp->url_path_ptr, comp->url_path_length, path_off, path_len);
  if (query_len) ok &= emit(comp->extra_info_ptr, comp->extra_info_length, query_off, query_len);

  return ok ? 1 : 0;
}

// XHttpCrackUrlW(caller, url, url_length, flags, components) -> BOOL.
// Wide twin of the above, sharing the component struct (only the strings differ
// in width; lengths stay character counts). The store's request builder cracks
// its composed URL through this one, and left stubbed it returned with the
// component pointers still null, which the caller then dereferenced.
u32 NetDll_XHttpCrackUrlW_entry(u32 caller, u32 url_guest, u32 url_length, u32 flags,
                                u32 components_guest) {
  auto* mem = REX_KERNEL_MEMORY();
  REXKRNL_INFO("XHttpCrackUrlW(url={:#x}, len={}, flags={:#x}, comp={:#x})", url_guest, url_length,
               flags, components_guest);
  if (!url_guest || !components_guest) {
    return 0;
  }
  auto* comp = mem->TranslateVirtual<rex::kernel::XHTTP_URL_COMPONENTS*>(components_guest);
  REXKRNL_INFO("  struct_size={} host(ptr={:#x} len={}) path(ptr={:#x} len={}) extra(ptr={:#x} len={})",
               uint32_t(comp->struct_size), uint32_t(comp->host_name_ptr),
               uint32_t(comp->host_name_length), uint32_t(comp->url_path_ptr),
               uint32_t(comp->url_path_length), uint32_t(comp->extra_info_ptr),
               uint32_t(comp->extra_info_length));
  if (comp->struct_size != sizeof(rex::kernel::XHTTP_URL_COMPONENTS)) {
    return 0;
  }

  const auto* src = mem->TranslateVirtual<const rex::be<uint16_t>*>(url_guest);
  std::u16string url;
  const uint32_t max_chars = url_length ? url_length : 2048u;
  for (uint32_t i = 0; i < max_chars; ++i) {
    const uint16_t c = src[i].get();
    if (!c) {
      break;
    }
    url.push_back(static_cast<char16_t>(c));
  }
  REXKRNL_INFO("XHttpCrackUrlW('{}')", rex::string::to_utf8(url));

  size_t scheme_off = 0, scheme_len = 0, host_off = 0, host_len = 0;
  size_t path_off = 0, path_len = 0, query_off = 0, query_len = 0;
  uint16_t port = 0;
  uint32_t scheme_enum = 0;

  size_t cursor = 0;
  const size_t scheme_sep = url.find(u"://");
  if (scheme_sep != std::u16string::npos) {
    scheme_off = 0;
    scheme_len = scheme_sep;
    cursor = scheme_sep + 3;
    if (url.compare(0, scheme_len, u"https") == 0) {
      scheme_enum = static_cast<uint32_t>(rex::kernel::X_INTERNET_SCHEME::HTTPS);
      port = 443;
    } else {
      scheme_enum = static_cast<uint32_t>(rex::kernel::X_INTERNET_SCHEME::HTTP);
      port = 80;
    }
  }

  const size_t host_start = cursor;
  const size_t path_start = url.find(u'/', cursor);
  const size_t hostport_end = (path_start == std::u16string::npos) ? url.size() : path_start;
  const std::u16string hostport = url.substr(host_start, hostport_end - host_start);
  const size_t colon = hostport.find(u':');
  host_off = host_start;
  if (colon != std::u16string::npos) {
    host_len = colon;
    uint32_t parsed = 0;
    for (size_t i = colon + 1; i < hostport.size(); ++i) {
      if (hostport[i] < u'0' || hostport[i] > u'9') {
        break;
      }
      parsed = parsed * 10 + static_cast<uint32_t>(hostport[i] - u'0');
    }
    port = static_cast<uint16_t>(parsed);
  } else {
    host_len = hostport.size();
  }

  if (path_start != std::u16string::npos) {
    const size_t q = url.find(u'?', path_start);
    if (q != std::u16string::npos) {
      path_off = path_start;
      path_len = q - path_start;
      query_off = q;
      query_len = url.size() - q;
    } else {
      path_off = path_start;
      path_len = url.size() - path_start;
    }
  }

  // No host means the caller's config never resolved. Report failure rather
  // than handing back components it will dereference blind.
  if (!host_len) {
    REXKRNL_WARN("XHttpCrackUrlW: no host in '{}'", rex::string::to_utf8(url));
    return 0;
  }

  // A caller that asks for no extra_info at all wants the query kept on the
  // path, which is how the store gets its arguments across: it requests host
  // and path only, then hands that path straight to XHttpOpenRequest.
  if (query_len && !comp->extra_info_ptr && !comp->extra_info_length) {
    path_len += query_len;
    query_len = 0;
  }

  comp->scheme = scheme_enum;
  comp->port = port;

  auto emit = [&](rex::be<uint32_t>& ptr_field, rex::be<uint32_t>& len_field, size_t off,
                  size_t len) -> bool {
    const uint32_t cur_ptr = ptr_field;
    const uint32_t cur_len = len_field;
    if (cur_ptr) {
      if (cur_len < len + 1) {
        len_field = static_cast<uint32_t>(len + 1);
        return false;
      }
      auto* dst = mem->TranslateVirtual<rex::be<uint16_t>*>(cur_ptr);
      for (size_t i = 0; i < len; ++i) {
        dst[i] = static_cast<uint16_t>(url[off + i]);
      }
      dst[len] = 0;
      len_field = static_cast<uint32_t>(len);
    } else if (cur_len) {
      ptr_field = url_guest + static_cast<uint32_t>(off * sizeof(char16_t));
      len_field = static_cast<uint32_t>(len);
    }
    return true;
  };

  bool ok = true;
  if (scheme_len) ok &= emit(comp->scheme_ptr, comp->scheme_length, scheme_off, scheme_len);
  if (host_len) ok &= emit(comp->host_name_ptr, comp->host_name_length, host_off, host_len);
  if (path_len) ok &= emit(comp->url_path_ptr, comp->url_path_length, path_off, path_len);
  if (query_len) ok &= emit(comp->extra_info_ptr, comp->extra_info_length, query_off, query_len);

  REXKRNL_INFO("XHttpCrackUrlW -> {} (host_len={}, path_len={}, port={})", ok, host_len, path_len,
               port);
  return ok ? 1 : 0;
}

}  // namespace xam
}  // namespace kernel
}  // namespace rex

REX_EXPORT(__imp__NetDll_XHttpStartup, rex::kernel::xam::NetDll_XHttpStartup_entry)
REX_EXPORT(__imp__NetDll_XHttpShutdown, rex::kernel::xam::NetDll_XHttpShutdown_entry)
REX_EXPORT(__imp__NetDll_XHttpOpen, rex::kernel::xam::NetDll_XHttpOpen_entry)
REX_EXPORT(__imp__NetDll_XHttpConnect, rex::kernel::xam::NetDll_XHttpConnect_entry)
REX_EXPORT(__imp__NetDll_XHttpOpenRequest, rex::kernel::xam::NetDll_XHttpOpenRequest_entry)
REX_EXPORT(__imp__NetDll_XHttpOpenRequestUsingMemory,
           rex::kernel::xam::NetDll_XHttpOpenRequestUsingMemory_entry)
REX_EXPORT(__imp__NetDll_XHttpSetStatusCallback,
           rex::kernel::xam::NetDll_XHttpSetStatusCallback_entry)
REX_EXPORT(__imp__NetDll_XHttpSendRequest, rex::kernel::xam::NetDll_XHttpSendRequest_entry)
REX_EXPORT(__imp__NetDll_XHttpReceiveResponse,
           rex::kernel::xam::NetDll_XHttpReceiveResponse_entry)
REX_EXPORT(__imp__NetDll_XHttpReadData, rex::kernel::xam::NetDll_XHttpReadData_entry)
REX_EXPORT(__imp__NetDll_XHttpQueryHeaders, rex::kernel::xam::NetDll_XHttpQueryHeaders_entry)
REX_EXPORT(__imp__NetDll_XHttpDoWork, rex::kernel::xam::NetDll_XHttpDoWork_entry)
REX_EXPORT(__imp__NetDll_XHttpCloseHandle, rex::kernel::xam::NetDll_XHttpCloseHandle_entry)
REX_EXPORT(__imp__NetDll_XHttpQueryOption, rex::kernel::xam::NetDll_XHttpQueryOption_entry)
REX_EXPORT(__imp__NetDll_XHttpSetOption, rex::kernel::xam::NetDll_XHttpSetOption_entry)

REX_EXPORT(__imp__XampXAuthStartup, rex::kernel::xam::XampXAuthStartup_entry)
REX_EXPORT(__imp__XampXAuthShutdown, rex::kernel::xam::XampXAuthShutdown_entry)
REX_EXPORT(__imp__XampXAuthGetTitleBuffer, rex::kernel::xam::XampXAuthGetTitleBuffer_entry)
REX_EXPORT(__imp__XampXAuthIsLocalSocketAllowed,
           rex::kernel::xam::XampXAuthIsLocalSocketAllowed_entry)

REX_EXPORT(__imp__XamGetServiceEndpoint, rex::kernel::xam::XamGetServiceEndpoint_entry)
REX_EXPORT(__imp__XamGetToken, rex::kernel::xam::XamGetToken_entry)
REX_EXPORT(__imp__NetDll_XHttpCrackUrl, rex::kernel::xam::NetDll_XHttpCrackUrl_entry)
REX_EXPORT(__imp__NetDll_XHttpCrackUrlW, rex::kernel::xam::NetDll_XHttpCrackUrlW_entry)
