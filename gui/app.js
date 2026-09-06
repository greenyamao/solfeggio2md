/**
 * Solfeggio OCR Workbench — Master Application Logic
 * Strict Zero-Scroll UI, Keyboard Hotkey Engine, OMR Inspector,
 * and Book Markdown Renderer.
 */

(function () {
  'use strict';

  // State
  const state = {
    pages: [],
    currentPageIndex: 0,
    pageData: null,
    activeMode: 1,
    selectedCropIndex: 0,
    activeCodeTab: 'abc',
    isRawMarkdownView: false,
    pollInterval: null
  };

  // DOM Elements Cache
  const DOM = {
    // Header
    btnPrevPage: document.getElementById('btn-prev-page'),
    btnNextPage: document.getElementById('btn-next-page'),
    pageSelect: document.getElementById('page-select'),
    pageCounter: document.getElementById('page-counter'),
    modeBtns: document.querySelectorAll('.btn-mode'),

    // View Panels
    panels: {
      1: document.getElementById('view-mask'),
      2: document.getElementById('view-normalization'),
      3: document.getElementById('view-omr'),
      4: document.getElementById('view-final')
    },

    // Mode 1: Mask
    scanDebugImg: document.getElementById('scan-debug-img'),
    scanMaskImg: document.getElementById('scan-mask-img'),
    badgeStavesCount: document.getElementById('badge-staves-count'),

    // Mode 2: Normalization
    stavesSubbar2: document.getElementById('staves-subbar-2'),
    normRawImg: document.getElementById('norm-raw-img'),
    normDeskewImg: document.getElementById('norm-deskew-img'),
    normAngleBadge: document.getElementById('norm-angle-badge'),
    normRawDim: document.getElementById('norm-raw-dim'),
    toggleGuidelines: document.getElementById('toggle-guidelines'),
    guidelinesLayer: document.getElementById('guidelines-layer'),
    normStaffId: document.getElementById('norm-staff-id'),
    normStaffClass: document.getElementById('norm-staff-class'),
    normStaffRes: document.getElementById('norm-staff-res'),
    normStaffStatus: document.getElementById('norm-staff-status'),

    // Mode 3: OMR
    stavesSubbar3: document.getElementById('staves-subbar-3'),
    omrCropImg: document.getElementById('omr-crop-img'),
    omrModelBadge: document.getElementById('omr-model-badge'),
    btnRerunOmr: document.getElementById('btn-rerun-omr'),
    btnTabAbc: document.getElementById('btn-tab-abc'),
    btnTabKern: document.getElementById('btn-tab-kern'),
    btnCopyCode: document.getElementById('btn-copy-code'),
    omrCodeBox: document.getElementById('omr-code-box'),
    omrCopyStatus: document.getElementById('omr-copy-status'),

    // Mode 4: Book Final
    finalScanImg: document.getElementById('final-scan-img'),
    bookRenderedContent: document.getElementById('book-rendered-content'),
    bookRawEditor: document.getElementById('book-raw-editor'),
    btnToggleMdRaw: document.getElementById('btn-toggle-md-raw'),
    btnSaveMd: document.getElementById('btn-save-md'),

    // Statusbar
    vramStatusPill: document.getElementById('vram-status-pill'),
    vramText: document.getElementById('vram-text'),
    deviceNameText: document.getElementById('device-name-text'),
    pageSummaryText: document.getElementById('page-summary-text')
  };

  // Configure Marked with custom music blocks
  function setupMarkedRenderer() {
    if (typeof marked === 'undefined') return;

    const renderer = new marked.Renderer();
    const originalCode = renderer.code.bind(renderer);

    renderer.code = function (code, lang) {
      if (lang === 'abc') {
        const lines = code.trim().split('\n');
        let title = 'Музыкальный фрагмент';
        let key = 'C';
        let meter = '4/4';

        lines.forEach(l => {
          if (l.startsWith('T:')) title = l.substring(2).trim();
          if (l.startsWith('K:')) key = l.substring(2).trim();
          if (l.startsWith('M:')) meter = l.substring(2).trim();
        });

        return `
          <div class="music-card">
            <div class="music-card-header">
              <span class="music-card-title">🎼 ${title}</span>
              <div>
                <span class="badge badge-staff">Ключ: ${key}</span>
                <span class="badge badge-grand">Метр: ${meter}</span>
              </div>
            </div>
            <pre class="music-card-code">${code}</pre>
          </div>
        `;
      }
      return originalCode(code, lang);
    };

    marked.setOptions({ renderer: renderer });
  }

  // API Call Wrapper (supports REST fetch and pywebview window.pywebview.api)
  async function fetchAPI(endpoint, options = {}) {
    try {
      const response = await fetch(endpoint, options);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return await response.json();
    } catch (err) {
      console.warn(`Fetch ${endpoint} error:`, err);
      // Fallback to pywebview API if available
      if (window.pywebview && window.pywebview.api) {
        const method = endpoint.replace('/api/', '');
        if (typeof window.pywebview.api[method] === 'function') {
          return await window.pywebview.api[method](options.body ? JSON.parse(options.body) : undefined);
        }
      }
      throw err;
    }
  }

  // Load Pages List
  async function loadPages() {
    try {
      const data = await fetchAPI('/api/pages');
      state.pages = data.pages || [];

      DOM.pageSelect.innerHTML = '';
      state.pages.forEach((p, idx) => {
        const opt = document.createElement('option');
        opt.value = p.page_id;
        opt.textContent = `${idx + 1}. ${p.title} (${p.crops_count} станов)`;
        DOM.pageSelect.appendChild(opt);
      });

      if (state.pages.length > 0) {
        state.currentPageIndex = 0;
        await selectPage(state.pages[0].page_id);
      }
    } catch (err) {
      DOM.pageSummaryText.textContent = 'Ошибка загрузки списка страниц: ' + err.message;
    }
  }

  // Select and Load Page Data
  async function selectPage(pageId) {
    const idx = state.pages.findIndex(p => p.page_id === pageId);
    if (idx !== -1) {
      state.currentPageIndex = idx;
    }

    DOM.pageSelect.value = pageId;
    DOM.pageCounter.textContent = `${state.currentPageIndex + 1} / ${state.pages.length}`;
    DOM.pageSummaryText.textContent = `Загрузка данных страницы ${pageId}...`;

    try {
      const data = await fetchAPI(`/api/page/${pageId}`);
      state.pageData = data;
      state.selectedCropIndex = 0;

      updateHardwareStatus(data.hardware);
      renderActiveView();
      DOM.pageSummaryText.textContent = `${data.title} • Станов: ${data.crops.length}`;
    } catch (err) {
      DOM.pageSummaryText.textContent = `Ошибка загрузки страницы ${pageId}: ${err.message}`;
    }
  }

  // Mode Switcher
  function switchMode(modeNum) {
    state.activeMode = modeNum;

    // Update Mode Buttons
    DOM.modeBtns.forEach(btn => {
      const mode = parseInt(btn.dataset.mode, 10);
      if (mode === modeNum) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    // Update Panels Visibility
    Object.keys(DOM.panels).forEach(key => {
      const panel = DOM.panels[key];
      if (parseInt(key, 10) === modeNum) {
        panel.classList.add('active');
      } else {
        panel.classList.remove('active');
      }
    });

    renderActiveView();
  }

  // Master Render Dispatcher
  function renderActiveView() {
    if (!state.pageData) return;

    switch (state.activeMode) {
      case 1:
        renderMode1();
        break;
      case 2:
        renderMode2();
        break;
      case 3:
        renderMode3();
        break;
      case 4:
        renderMode4();
        break;
    }
  }

  // Render Mode 1: Scan vs Mask
  function renderMode1() {
    const d = state.pageData;
    DOM.scanDebugImg.src = d.debug_url || '';
    DOM.scanMaskImg.src = d.mask_url || '';
    DOM.badgeStavesCount.textContent = `${d.crops.length} станов`;
  }

  // Render Mode 2: Normalization & Deskew
  function renderMode2() {
    const crops = state.pageData.crops || [];
    DOM.stavesSubbar2.innerHTML = '';

    if (crops.length === 0) {
      DOM.normRawImg.src = '';
      DOM.normDeskewImg.src = '';
      DOM.normStaffId.textContent = 'Нет станов';
      return;
    }

    if (state.selectedCropIndex >= crops.length) {
      state.selectedCropIndex = 0;
    }

    // Build Sub-bar Tabs
    crops.forEach((crop, idx) => {
      const chip = document.createElement('button');
      chip.className = `staff-chip ${idx === state.selectedCropIndex ? 'active' : ''}`;
      chip.innerHTML = `<span>#${crop.index} ${crop.class}</span> <span class="badge badge-grand">${crop.skew_angle > 0 ? '+' : ''}${crop.skew_angle}°</span>`;
      chip.addEventListener('click', () => {
        state.selectedCropIndex = idx;
        renderMode2();
      });
      DOM.stavesSubbar2.appendChild(chip);
    });

    const activeCrop = crops[state.selectedCropIndex];
    DOM.normRawImg.src = activeCrop.raw_url;
    DOM.normDeskewImg.src = activeCrop.deskew_url;
    DOM.normAngleBadge.textContent = `Угол доворота: ${activeCrop.skew_angle > 0 ? '+' : ''}${activeCrop.skew_angle}°`;
    DOM.normRawDim.textContent = `${activeCrop.width} × ${activeCrop.height} px`;

    DOM.normStaffId.textContent = `#${activeCrop.index} (${activeCrop.id})`;
    DOM.normStaffClass.textContent = activeCrop.class;
    DOM.normStaffRes.textContent = `${activeCrop.width} × ${activeCrop.height} px`;
    DOM.normStaffStatus.textContent = Math.abs(activeCrop.skew_angle) < 0.05 ? 'Идеально горизонтально' : 'Выровнен jdeskew';
  }

  // Render Mode 3: OMR Verification
  function renderMode3() {
    const crops = state.pageData.crops || [];
    DOM.stavesSubbar3.innerHTML = '';

    if (crops.length === 0) {
      DOM.omrCropImg.src = '';
      DOM.omrCodeBox.textContent = 'На странице не обнаружено нотных станов.';
      return;
    }

    if (state.selectedCropIndex >= crops.length) {
      state.selectedCropIndex = 0;
    }

    crops.forEach((crop, idx) => {
      const chip = document.createElement('button');
      chip.className = `staff-chip ${idx === state.selectedCropIndex ? 'active' : ''}`;
      chip.innerHTML = `<span>#${crop.index} ${crop.class}</span>`;
      chip.addEventListener('click', () => {
        state.selectedCropIndex = idx;
        renderMode3();
      });
      DOM.stavesSubbar3.appendChild(chip);
    });

    const activeCrop = crops[state.selectedCropIndex];
    DOM.omrCropImg.src = activeCrop.deskew_url || activeCrop.raw_url;
    DOM.omrModelBadge.textContent = activeCrop.model_used;

    const code = state.activeCodeTab === 'abc' ? activeCrop.abc : (activeCrop.kern || 'Нет kern данных');
    DOM.omrCodeBox.textContent = code;
  }

  // Render Mode 4: Book Final
  function renderMode4() {
    const d = state.pageData;
    DOM.finalScanImg.src = d.original_url || d.debug_url;

    if (state.isRawMarkdownView) {
      DOM.bookRenderedContent.style.display = 'none';
      DOM.bookRawEditor.style.display = 'block';
      DOM.bookRawEditor.value = d.markdown;
      DOM.btnToggleMdRaw.textContent = '👁️ Предпросмотр книги';
    } else {
      DOM.bookRawEditor.style.display = 'none';
      DOM.bookRenderedContent.style.display = 'block';
      DOM.btnToggleMdRaw.textContent = '📝 Исходный текст';

      if (typeof marked !== 'undefined') {
        DOM.bookRenderedContent.innerHTML = marked.parse(d.markdown);
      } else {
        DOM.bookRenderedContent.innerHTML = `<pre>${d.markdown}</pre>`;
      }
    }
  }

  // Update Statusbar VRAM Pill
  function updateHardwareStatus(hw) {
    if (!hw) return;
    DOM.deviceNameText.textContent = `Устройство: ${hw.device_name}`;
    DOM.vramText.textContent = `VRAM: ${hw.vram_allocated_mb} MB (${hw.status_text})`;

    if (hw.is_safe_for_vlm) {
      DOM.vramStatusPill.className = 'vram-pill vram-safe';
    } else {
      DOM.vramStatusPill.className = 'vram-pill vram-busy';
    }
  }

  // Keyboard Shortcuts Handler
  function setupKeyboardNavigation() {
    window.addEventListener('keydown', (e) => {
      // Don't intercept if user is typing in textarea or contenteditable
      const targetTag = e.target.tagName ? e.target.tagName.toLowerCase() : '';
      if (targetTag === 'input' || targetTag === 'textarea' || e.target.isContentEditable) {
        return;
      }

      // Mode Switchers 1, 2, 3, 4
      if (e.key === '1') {
        e.preventDefault();
        switchMode(1);
      } else if (e.key === '2') {
        e.preventDefault();
        switchMode(2);
      } else if (e.key === '3') {
        e.preventDefault();
        switchMode(3);
      } else if (e.key === '4') {
        e.preventDefault();
        switchMode(4);
      }

      // Page Navigation (ArrowLeft / ArrowRight or A / D)
      else if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {
        e.preventDefault();
        prevPage();
      } else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {
        e.preventDefault();
        nextPage();
      }

      // Staves Navigation (ArrowUp / ArrowDown in Mode 2 and 3)
      else if (e.key === 'ArrowUp') {
        e.preventDefault();
        prevCrop();
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        nextCrop();
      }
    });
  }

  function prevPage() {
    if (state.currentPageIndex > 0) {
      state.currentPageIndex--;
      selectPage(state.pages[state.currentPageIndex].page_id);
    }
  }

  function nextPage() {
    if (state.currentPageIndex < state.pages.length - 1) {
      state.currentPageIndex++;
      selectPage(state.pages[state.currentPageIndex].page_id);
    }
  }

  function prevCrop() {
    if (!state.pageData || !state.pageData.crops) return;
    if (state.selectedCropIndex > 0) {
      state.selectedCropIndex--;
      if (state.activeMode === 2) renderMode2();
      if (state.activeMode === 3) renderMode3();
    }
  }

  function nextCrop() {
    if (!state.pageData || !state.pageData.crops) return;
    if (state.selectedCropIndex < state.pageData.crops.length - 1) {
      state.selectedCropIndex++;
      if (state.activeMode === 2) renderMode2();
      if (state.activeMode === 3) renderMode3();
    }
  }

  // Setup Event Listeners
  function setupEventListeners() {
    // Mode Buttons Click
    DOM.modeBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const mode = parseInt(btn.dataset.mode, 10);
        switchMode(mode);
      });
    });

    // Page Navigation Buttons
    DOM.btnPrevPage.addEventListener('click', prevPage);
    DOM.btnNextPage.addEventListener('click', nextPage);
    DOM.pageSelect.addEventListener('change', (e) => {
      selectPage(e.target.value);
    });

    // Toggle Guidelines in Mode 2
    DOM.toggleGuidelines.addEventListener('change', (e) => {
      DOM.guidelinesLayer.style.display = e.target.checked ? 'block' : 'none';
    });

    // OMR Code Tabs in Mode 3
    DOM.btnTabAbc.addEventListener('click', () => {
      state.activeCodeTab = 'abc';
      DOM.btnTabAbc.classList.add('active');
      DOM.btnTabKern.classList.remove('active');
      renderMode3();
    });

    DOM.btnTabKern.addEventListener('click', () => {
      state.activeCodeTab = 'kern';
      DOM.btnTabKern.classList.add('active');
      DOM.btnTabAbc.classList.remove('active');
      renderMode3();
    });

    // Copy Code Button
    DOM.btnCopyCode.addEventListener('click', () => {
      const text = DOM.omrCodeBox.textContent;
      navigator.clipboard.writeText(text).then(() => {
        DOM.omrCopyStatus.style.display = 'inline';
        setTimeout(() => {
          DOM.omrCopyStatus.style.display = 'none';
        }, 2000);
      });
    });

    // Live Re-run OMR Button
    DOM.btnRerunOmr.addEventListener('click', async () => {
      if (!state.pageData || !state.pageData.crops) return;
      const crop = state.pageData.crops[state.selectedCropIndex];
      if (!crop) return;

      DOM.btnRerunOmr.disabled = true;
      DOM.btnRerunOmr.innerHTML = '<span>⏳ Распознавание...</span>';

      try {
        const result = await fetchAPI('/api/transcribe_crop', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            page_id: state.pageData.page_id,
            crop_stem: crop.crop_stem
          })
        });

        if (result.status === 'success') {
          crop.abc = result.abc;
          crop.kern = result.kern;
          crop.model_used = result.model_used;
          renderMode3();
        } else {
          alert('Ошибка OMR: ' + (result.message || 'Неизвестная ошибка'));
        }
      } catch (err) {
        alert('Ошибка связи с OMR сервером: ' + err.message);
      } finally {
        DOM.btnRerunOmr.disabled = false;
        DOM.btnRerunOmr.innerHTML = '<span>⚡ Перераспознать стан</span>';
      }
    });

    // Mode 4: Toggle Raw Markdown vs Rendered Book
    DOM.btnToggleMdRaw.addEventListener('click', () => {
      state.isRawMarkdownView = !state.isRawMarkdownView;
      if (!state.isRawMarkdownView) {
        // Sync edits from textarea back to state
        state.pageData.markdown = DOM.bookRawEditor.value;
      }
      renderMode4();
    });

    // Mode 4: Save Markdown
    DOM.btnSaveMd.addEventListener('click', async () => {
      if (!state.pageData) return;
      const content = state.isRawMarkdownView ? DOM.bookRawEditor.value : state.pageData.markdown;

      try {
        const res = await fetchAPI('/api/save_markdown', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            page_id: state.pageData.page_id,
            markdown: content
          })
        });

        if (res.status === 'success') {
          DOM.btnSaveMd.textContent = '✅ Сохранено!';
          setTimeout(() => {
            DOM.btnSaveMd.textContent = '💾 Сохранить';
          }, 2000);
        }
      } catch (err) {
        alert('Ошибка сохранения: ' + err.message);
      }
    });
  }

  // Hardware Status Poller
  function startHardwarePoller() {
    state.pollInterval = setInterval(async () => {
      try {
        const hw = await fetchAPI('/api/hardware');
        updateHardwareStatus(hw);
      } catch (e) {
        // Silent poll error
      }
    }, 4000);
  }

  // App Entrypoint
  async function init() {
    setupMarkedRenderer();
    setupKeyboardNavigation();
    setupEventListeners();
    await loadPages();
    startHardwarePoller();
  }

  // DOM Ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
