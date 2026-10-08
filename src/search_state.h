// search_state.h: Ctrl+F box state shared between the controller and the XUI hooks.
#pragma once

namespace ae_search {

// Replace the box state; lines are the result rows, the first one is the match count.
void SetSearchOverlay(bool open, const char* query, const char* const* lines, int line_count);

// Heading text for the possessed XUI label, returns the byte length written (0 when closed).
int GetSearchStatusLine(char* utf8, int cap);

// Filter armed after Enter, so the heading reads "Filter: q" until cleared.
void SetSearchApplied(bool applied);

// Controller -> frame tick bridge for the guest's catalog rebuild.
void RequestCatalogRebuild();
bool ConsumeCatalogRebuildRequest();

// The in-game heading label currently carries the status line.
void SetSearchLabelPossessed(bool possessed);

}  // namespace ae_search
