// 顶栏的「版式」菜单：切换 <html data-site-style>，记在 localStorage，所有页面生效
(function () {
  var KEY = "ai-learn.style";
  var styles = window.AI_LEARN_STYLES || {};
  var ICON = '<svg viewBox="0 0 24 24"><path d="M3 3h8v8H3V3m2 2v4h4V5H5m8-2h8v8h-8V3m2 2v4h4V5h-4M3 13h8v8H3v-8m2 2v4h4v-4H5m8-2h8v8h-8v-8m2 2v4h4v-4h-4"/></svg>';

  // 无标签版式下左侧栏列出全部领域，打开页面时把当前条目滚到侧栏可视区
  function revealActive() {
    if (current() === "docs") return;
    var link = document.querySelector(".md-nav--primary .md-nav__link--active");
    var wrap = link && link.closest(".md-sidebar__scrollwrap");
    if (!wrap) return;
    var offset = link.getBoundingClientRect().top - wrap.getBoundingClientRect().top;
    wrap.scrollTop += offset - wrap.clientHeight / 3;
  }

  function current() { return document.documentElement.getAttribute("data-site-style"); }

  function apply(name) {
    if (!styles[name]) return;
    document.documentElement.setAttribute("data-site-style", name);
    try { localStorage.setItem(KEY, name); } catch (e) {}
    var font = styles[name].font;
    if (font && !document.querySelector('link[href="' + font + '"]')) {
      var l = document.createElement("link");
      l.rel = "stylesheet"; l.href = font;
      document.head.appendChild(l);
    }
    render();
    revealActive();
  }

  var root, label, menu;
  function render() {
    if (!root) return;
    label.textContent = (styles[current()] || {}).label || "版式";
    Array.prototype.forEach.call(menu.querySelectorAll("button"), function (b) {
      b.setAttribute("aria-checked", b.dataset.style === current() ? "true" : "false");
    });
  }

  function mount() {
    var anchor = document.querySelector(".md-header__inner .md-header__option")
      || document.querySelector(".md-header__inner .md-search");
    if (!anchor || document.querySelector(".ai-style")) return;
    root = document.createElement("div");
    root.className = "ai-style";
    root.innerHTML = '<button class="ai-style__btn" type="button" title="切换版式" aria-haspopup="true">'
      + ICON + '<span class="ai-style__label"></span></button><ul class="ai-style__menu" role="menu"></ul>';
    label = root.querySelector(".ai-style__label");
    menu = root.querySelector(".ai-style__menu");
    Object.keys(styles).forEach(function (name) {
      var li = document.createElement("li");
      li.innerHTML = '<button type="button" role="menuitemradio" data-style="' + name + '">'
        + styles[name].label + "<small>" + styles[name].desc + "</small></button>";
      menu.appendChild(li);
    });
    root.querySelector(".ai-style__btn").addEventListener("click", function (e) {
      e.stopPropagation();
      root.classList.toggle("ai-style--open");
    });
    menu.addEventListener("click", function (e) {
      var b = e.target.closest("button[data-style]");
      if (b) { apply(b.dataset.style); root.classList.remove("ai-style--open"); }
    });
    document.addEventListener("click", function () { root.classList.remove("ai-style--open"); });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") root.classList.remove("ai-style--open"); });
    anchor.parentNode.insertBefore(root, anchor);
    render();
    revealActive();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount);
  else mount();
})();
