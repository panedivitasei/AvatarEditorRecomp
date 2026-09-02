// catalog_search.h: the Ctrl+F finder's input side. Typed text becomes the
// catalog filter and the heading's status line; while the box is open the
// listener sits topmost so keys never reach the pad drivers.
#pragma once

#include <string>

#include <rex/ui/window.h>
#include <rex/ui/window_listener.h>

namespace ae_search {

class Controller : public rex::ui::WindowInputListener {
 public:
  // Registers on the window above every other listener.
  void Attach(rex::ui::Window* window);
  void Detach();

  bool open() const { return open_; }

  void OnKeyDown(rex::ui::KeyEvent& e) override;
  void OnKeyUp(rex::ui::KeyEvent& e) override;
  void OnKeyChar(rex::ui::KeyEvent& e) override;

 private:
  void Toggle();
  void Apply(bool clear);
  void Push();

  rex::ui::Window* window_ = nullptr;
  bool open_ = false;
  bool applied_ = false;
  bool games_mode_ = false;  // the box was opened on the Game Styles list
  std::string query_;
};

Controller& Get();

// The store's Game Styles list, reported per frame by the title tick. Its
// filter rides on the FindGames query and Enter has the tick re-enter the page.
void SetGamesListOpen(bool open);
bool GamesListOpen();
bool ConsumeGamesReloadRequest();

}  // namespace ae_search
