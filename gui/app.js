/**
 * Solfeggio OCR Studio — Master Application Logic
 * Dual-Workspace Architecture: Batch Pipeline Dashboard + Visual Inspection Workbench.
 * Real-Time HUD Metrics, Queue Management, LM Studio Diagnostics, and Zero-Scroll Layout.
 */

(function () {
  'use strict';

  // Master State
  const state = {
    activeWorkspace: 'dashboard', // 'dashboard' | 'workbench'
    
    // Batch Pipeline State
    queue: [],
    pipelineMetrics: null,
    statusPollTimer: null,
    
    // Inspection Workbench State
    pages: [],
    currentPageIndex: 0,
    pageData: null,
    activeMode: 1,
    selectedCropIndex: 0,
    activeCodeTab: 'abc',
    isRawMarkdownView: false,
    showYoloDebugMode1: true,
    finalLeftView: 'orig'
  };

  // DOM Elements Cache
  const DOM = {
    // Workspaces Switcher
    btnWsDashboard: document.getElementById('btn-ws-dashboard'),
    btnWsWorkbench: document.getElementById('btn-ws-workbench'),
    wsDashboard: document.getElementById('workspace-dashboard'),
    wsWorkbench: document.getElementById('workspace-workbench'),
    workbenchSubmodes: document.getElementById('workbench-submodes'),

    // HUD Metrics Elements
    hudQueueMetric: document.getElementById('hud-queue-metric'),
    hudQueueBar: document.getElementById('hud-queue-bar'),
    hudTimeMetric: document.getElementById('hud-time-metric'),
    hudPhaseName: document.getElementById('hud-phase-name'),
    hudPhaseBar: document.getElementById('hud-phase-bar'),
    hudItemDetail: document.getElementById('hud-item-detail'),
    hudP1Bar: document.getElementById('hud-p1-bar'),
    hudP1Badge: document.getElementById('hud-p1-badge'),
    hudP2Bar: document.getElementById('hud-p2-bar'),
    hudP2Badge: document.getElementById('hud-p2-badge'),
    hudP3Bar: document.getElementById('hud-p3-bar'),
    hudP3Badge: document.getElementById('hud-p3-badge'),
    hudP4Bar: document.getElementById('hud-p4-bar'),
    hudP4Badge: document.getElementById('hud-p4-badge'),

    // Pipeline Controls
    btnPipelineStart: document.getElementById('btn-pipeline-start'),
    btnPipelinePause: document.getElementById('btn-pipeline-pause'),
    btnPipelineStop: document.getElementById('btn-pipeline-stop'),
    btnQueueClear: document.getElementById('btn-queue-clear'),

    // Queue & Dropzone
    pdfDropzone: document.getElementById('pdf-dropzone'),
    pdfFileInput: document.getElementById('pdf-file-input'),
    queueCountBadge: document.getElementById('queue-count-badge'),
    queueTableBody: document.getElementById('queue-table-body'),
    livePreviewImg: document.getElementById('live-preview-img'),
    liveEmptyBanner: document.getElementById('live-empty-banner'),
    liveBookBadge: document.getElementById('live-book-badge'),
    logConsoleBox: document.getElementById('log-console-box'),

    // Workbench Navigation & Modes
    btnPrevPage: document.getElementById('btn-prev-page'),
    btnNextPage: document.getElementById('btn-next-page'),
    pageSelect: document.getElementById('page-select'),
    pageCounter: document.getElementById('page-counter'),
    btnRefreshPages: document.getElementById('btn-refresh-pages'),
    modeBtns: document.querySelectorAll('.btn-mode'),

    // Workbench Panels
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
    mode1LeftTitle: document.getElementById('mode1-left-title'),
    badgeSpreadInfo: document.getElementById('badge-spread-info'),
    btnToggleScanMode: document.getElementById('btn-toggle-scan-mode'),

    // Mode 2: Normalization
    stavesSubbar2: document.getElementById('staves-subbar-2'),
    normRawImg: document.getElementById('norm-raw-img'),
    normDeskewImg: document.getElementById('norm-deskew-img'),
    normDeskewTitle: document.getElementById('norm-deskew-title'),
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
    btnSynthPlay: document.getElementById('btn-synth-play'),
    omrCodeBox: document.getElementById('omr-code-box'),
    omrCopyStatus: document.getElementById('omr-copy-status'),
    omrSheetCanvas: document.getElementById('omr-sheet-canvas'),
    omrSheetViewport: document.getElementById('omr-sheet-viewport'),
    omrSheetDiagnostics: document.getElementById('omr-sheet-diagnostics'),
    omrRenderTiming: document.getElementById('omr-render-timing'),
    omrSheetEngineBadge: document.getElementById('omr-sheet-engine-badge'),
    btnToggleScoreTheme: document.getElementById('btn-toggle-score-theme'),

    // Mode 4: Book Final
    finalScanImg: document.getElementById('final-scan-img'),
    finalLeftTitle: document.getElementById('final-left-title'),
    finalLeftBadge: document.getElementById('final-left-badge'),
    btnFinalViewOrig: document.getElementById('btn-final-view-orig'),
    btnFinalViewMask: document.getElementById('btn-final-view-mask'),
    btnFinalViewDebug: document.getElementById('btn-final-view-debug'),
    bookRenderedContent: document.getElementById('book-rendered-content'),
    bookRawEditor: document.getElementById('book-raw-editor'),
    btnToggleMdRaw: document.getElementById('btn-toggle-md-raw'),
    btnSaveMd: document.getElementById('btn-save-md'),

    // Statusbar
    vramStatusPill: document.getElementById('vram-status-pill'),
    vramText: document.getElementById('vram-text'),
    deviceNameText: document.getElementById('device-name-text'),
    pageSummaryText: document.getElementById('page-summary-text'),

    // Settings Modal
    btnOpenSettings: document.getElementById('btn-open-settings'),
    modalSettings: document.getElementById('modal-settings'),
    btnCloseSettings: document.getElementById('btn-close-settings'),
    btnCancelSettings: document.getElementById('btn-cancel-settings'),
    btnSaveSettings: document.getElementById('btn-save-settings'),
    btnTestLmstudio: document.getElementById('btn-test-lmstudio'),
    lmstudioTestResult: document.getElementById('lmstudio-test-result'),

    // Settings Fields
    cfgLmHost: document.getElementById('cfg-lm-host'),
    cfgLmPort: document.getElementById('cfg-lm-port'),
    cfgLmModel: document.getElementById('cfg-lm-model'),
    cfgLmTemp: document.getElementById('cfg-lm-temp'),
    cfgLmTokens: document.getElementById('cfg-lm-tokens'),
    cfgQwenContext: document.getElementById('cfg-qwen-context'),
    cfgQwenBatch: document.getElementById('cfg-qwen-batch'),
    cfgFlashAtt: document.getElementById('cfg-flash-att'),
    cfgDpi: document.getElementById('cfg-dpi'),
    cfgSmtTokens: document.getElementById('cfg-smt-tokens'),
    cfgSmtDevice: document.getElementById('cfg-smt-device'),
    cfgOverwrite: document.getElementById('cfg-overwrite'),
    cfgSystemPrompt: document.getElementById('cfg-system-prompt')
  };

  // API Call Wrapper
  async function api(endpoint, options = {}) {
    const res = await fetch(endpoint, options);
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    return await res.json();
  }

  // ========================================================================
  // Workspace Switching
  // ========================================================================
  function switchWorkspace(target) {
    state.activeWorkspace = target;

    if (target === 'dashboard') {
      DOM.btnWsDashboard.classList.add('active');
      DOM.btnWsWorkbench.classList.remove('active');
      DOM.wsDashboard.classList.add('active');
      DOM.wsWorkbench.classList.remove('active');
      DOM.workbenchSubmodes.style.display = 'none';
      refreshQueue();
    } else {
      DOM.btnWsWorkbench.classList.add('active');
      DOM.btnWsDashboard.classList.remove('active');
      DOM.wsWorkbench.classList.add('active');
      DOM.wsDashboard.classList.remove('active');
      DOM.workbenchSubmodes.style.display = 'flex';
      loadWorkbenchPages(true);
      renderActiveWorkbenchMode();
    }
  }

  // ========================================================================
  // Batch Pipeline & Queue
  // ========================================================================
  async function refreshQueue() {
    try {
      const data = await api('/api/queue');
      state.queue = data.queue || [];
      renderQueueTable();
    } catch (e) {
      console.warn('Refresh queue failed:', e);
    }
  }

  function renderQueueTable() {
    DOM.queueCountBadge.textContent = `${state.queue.length} файлов`;
    DOM.queueTableBody.innerHTML = '';

    if (state.queue.length === 0) {
      DOM.queueTableBody.innerHTML = `
        <div class="queue-empty-placeholder">
          <span>Очередь пуста. Перетащите PDF файлы в зону выше.</span>
        </div>`;
      return;
    }

    state.queue.forEach((item) => {
      const row = document.createElement('div');
      row.className = 'queue-item-row';

      let badgeCls = 'badge-pending';
      let badgeTxt = 'Ожидает';
      if (item.status === 'processing') {
        badgeCls = 'badge-processing';
        badgeTxt = 'В обработке';
      } else if (item.status === 'completed') {
        badgeCls = 'badge-completed';
        badgeTxt = 'Готово';
      } else if (item.status === 'error') {
        badgeCls = 'badge-error';
        badgeTxt = 'Ошибка';
      }

      row.innerHTML = `
        <div class="queue-item-info">
          <span style="color: var(--text-muted); font-family: var(--font-mono); font-size: 12px; width: 24px;">#${item.id}</span>
          <div>
            <div class="queue-item-name">${item.name}</div>
            <div class="queue-item-meta">${item.pages} страниц • ${item.path}</div>
          </div>
        </div>
        <div class="queue-item-actions">
          <span class="badge ${badgeCls}">${badgeTxt}</span>
          <button class="btn-nav" data-remove-id="${item.id}" title="Убрать из очереди" style="color: var(--accent-rose); font-size: 12px;">✕</button>
        </div>
      `;

      row.querySelector('[data-remove-id]').addEventListener('click', async () => {
        await api('/api/queue/remove', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id: item.id })
        });
        await refreshQueue();
      });

      DOM.queueTableBody.appendChild(row);
    });
  }

  function updatePhaseBadge(barEl, badgeEl, phase, unitName) {
    if (!barEl || !badgeEl) return;
    if (!phase) {
      barEl.style.width = '0%';
      badgeEl.textContent = 'Ожидание';
      badgeEl.className = 'hud-phase-badge';
      return;
    }

    const pct = typeof phase.pct === 'number' ? Math.min(100, Math.max(0, phase.pct)) : 0;
    barEl.style.width = `${pct}%`;

    const status = phase.status || 'pending';
    badgeEl.className = `hud-phase-badge ${status}`;

    if (status === 'completed') {
      badgeEl.textContent = phase.total > 0 ? `100% (${phase.total} ${unitName})` : '100% Готово';
    } else if (status === 'skipped') {
      badgeEl.textContent = 'Пропущено';
    } else if (status === 'running') {
      badgeEl.textContent = phase.total > 0 ? `${phase.done}/${phase.total} ${unitName}` : `${pct.toFixed(0)}%`;
    } else {
      badgeEl.textContent = 'Ожидание';
    }
  }

  async function pollPipelineStatus() {
    try {
      const data = await api('/api/pipeline/status');
      state.pipelineMetrics = data;

      // Update HUD Master Queue Progress & Time
      const qPct = typeof data.queue_progress_pct === 'number' ? data.queue_progress_pct : 0.0;
      const curBook = data.current_book_index || 0;
      const totBooks = data.total_books || 0;
      const bookName = data.current_book_name ? ` • ${data.current_book_name}` : '';
      DOM.hudQueueMetric.textContent = `${qPct.toFixed(1)}% (Книг: ${curBook}/${totBooks}${bookName})`;
      if (DOM.hudQueueBar) DOM.hudQueueBar.style.width = `${Math.min(100, Math.max(0, qPct))}%`;

      if (DOM.hudPhaseName) DOM.hudPhaseName.textContent = data.current_phase_name || 'Готов к запуску';
      if (DOM.hudPhaseBar) DOM.hudPhaseBar.style.width = `${Math.min(100, data.book_progress_pct || 0)}%`;
      if (DOM.hudItemDetail) DOM.hudItemDetail.textContent = data.current_item_detail || 'Очередь ожидает команды';

      const elapsed = formatSeconds(data.elapsed_seconds);
      DOM.hudTimeMetric.textContent = `Прошло: ${elapsed}`;

      // Update 4 Dedicated Phase Bars & Badges
      const pp = data.phase_progress || {};
      updatePhaseBadge(DOM.hudP1Bar, DOM.hudP1Badge, pp.phase1, 'стр.');
      updatePhaseBadge(DOM.hudP2Bar, DOM.hudP2Badge, pp.phase2, 'стан.');
      updatePhaseBadge(DOM.hudP3Bar, DOM.hudP3Badge, pp.phase3, 'стр.');
      updatePhaseBadge(DOM.hudP4Bar, DOM.hudP4Badge, pp.phase4, 'стр.');

      // Auto-refresh queue every 2.5 seconds ONLY when on dashboard tab
      // so we don't contend for Python GIL and disk I/O while inspecting pages in workbench
      if (state.activeWorkspace === 'dashboard') {
        if (!state._lastQueueSync || (Date.now() - state._lastQueueSync > 2500)) {
          state._lastQueueSync = Date.now();
          refreshQueue();
        }
      }

      // Initial load of workbench pages if empty
      if (state.activeWorkspace === 'workbench' && state.pages.length === 0) {
        if (!state._lastPagesCheck || (Date.now() - state._lastPagesCheck > 2000)) {
          state._lastPagesCheck = Date.now();
          loadWorkbenchPages(false);
        }
      }

      // Update Buttons
      DOM.btnPipelineStart.disabled = data.is_running && !data.is_paused;
      DOM.btnPipelinePause.disabled = !data.is_running || data.is_paused;
      DOM.btnPipelineStop.disabled = !data.is_running;

      if (data.is_paused) {
        DOM.btnPipelineStart.querySelector('span').textContent = 'Продолжить';
      } else if (data.is_running) {
        DOM.btnPipelineStart.querySelector('span').textContent = 'В работе...';
      } else {
        DOM.btnPipelineStart.querySelector('span').textContent = 'Запустить конвейер';
      }

      // Log Appending
      if (data.last_log && data.last_log !== DOM.logConsoleBox.dataset.lastLog) {
        DOM.logConsoleBox.dataset.lastLog = data.last_log;
        const timestamp = new Date().toLocaleTimeString();
        DOM.logConsoleBox.textContent += `[${timestamp}] ${data.last_log}\n`;
        DOM.logConsoleBox.scrollTop = DOM.logConsoleBox.scrollHeight;
      }

      // Update Status Bar with Process-Isolated Telemetry
      const telem = data.process_telemetry || {};
      const cpuPct = telem.cpu_percent !== undefined ? telem.cpu_percent : (data.cpu_percent || 0);
      const ramMb = telem.ram_rss_mb !== undefined ? telem.ram_rss_mb : (data.ram_rss_mb || 0);
      const vramMb = telem.vram_allocated_mb !== undefined ? telem.vram_allocated_mb : (data.vram_allocated_mb || 0);
      const devName = telem.device_name || data.device_name || 'CPU';
      const mode = telem.runtime_mode || data.runtime_mode || 'CPU Mode';
      const isCuda = telem.cuda_available || false;

      DOM.deviceNameText.textContent = `[Процесс] CPU: ${cpuPct}% • RAM: ${ramMb} MB • ${devName}`;
      if (vramMb < 500) {
        DOM.vramStatusPill.className = 'vram-pill vram-safe';
        DOM.vramText.textContent = isCuda ? `VRAM: ${vramMb} MB (Safe for VLM)` : `VRAM: ${vramMb} MB (${mode})`;
      } else {
        DOM.vramStatusPill.className = 'vram-pill vram-busy';
        DOM.vramText.textContent = `VRAM: ${vramMb} MB (OMR в памяти)`;
      }
    } catch (e) {
      // Offline/quiet
    }
  }

  function formatSeconds(sec) {
    const s = Math.floor(sec || 0);
    const m = Math.floor(s / 60);
    const rem = s % 60;
    return `${m.toString().padStart(2, '0')}:${rem.toString().padStart(2, '0')}`;
  }

  // ========================================================================
  // Drag & Drop Handling
  // ========================================================================
  function setupDropzone() {
    DOM.pdfDropzone.addEventListener('click', () => {
      DOM.pdfFileInput.click();
    });

    // Prevent default window navigation when dragging files onto WebView
    window.addEventListener('dragover', (e) => e.preventDefault());
    window.addEventListener('drop', (e) => e.preventDefault());

    async function handleUploadFiles(files) {
      const pdfFiles = Array.from(files).filter(f => f.name.toLowerCase().endsWith('.pdf'));
      if (pdfFiles.length === 0) {
        alert('Пожалуйста, выберите файл в формате .pdf');
        return;
      }
      DOM.queueCountBadge.textContent = 'Загрузка...';
      for (const file of pdfFiles) {
        const formData = new FormData();
        formData.append('file', file);
        try {
          const resp = await fetch('/api/queue/upload', { method: 'POST', body: formData });
          if (!resp.ok) {
            console.error('Upload failed with status', resp.status);
          }
        } catch (err) {
          console.error('Upload error:', err);
        }
      }
      await refreshQueue();
    }

    DOM.pdfFileInput.addEventListener('change', async (e) => {
      await handleUploadFiles(e.target.files);
      DOM.pdfFileInput.value = '';
    });

    DOM.pdfDropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      DOM.pdfDropzone.classList.add('drag-over');
    });

    DOM.pdfDropzone.addEventListener('dragleave', () => {
      DOM.pdfDropzone.classList.remove('drag-over');
    });

    DOM.pdfDropzone.addEventListener('drop', async (e) => {
      e.preventDefault();
      DOM.pdfDropzone.classList.remove('drag-over');
      if (e.dataTransfer && e.dataTransfer.files) {
        await handleUploadFiles(e.dataTransfer.files);
      }
    });
  }

  // ========================================================================
  // Settings Modal & Diagnostics
  // ========================================================================
  async function openSettings() {
    try {
      const cfg = await api('/api/config');
      DOM.cfgLmHost.value = cfg.lm_host || '127.0.0.1';
      DOM.cfgLmPort.value = cfg.lm_port || '1234';
      DOM.cfgLmModel.value = cfg.lm_model || 'qwen/qwen3.5-9b';
      DOM.cfgLmTemp.value = cfg.lm_temperature !== undefined ? cfg.lm_temperature : 0.1;
      DOM.cfgLmTokens.value = cfg.lm_max_tokens || 8192;
      DOM.cfgQwenContext.value = cfg.qwen_context_length || 16196;
      DOM.cfgQwenBatch.value = cfg.qwen_eval_batch_size || 2048;
      DOM.cfgFlashAtt.checked = cfg.qwen_flash_attention !== false;
      DOM.cfgDpi.value = cfg.dpi || 200;
      DOM.cfgSmtTokens.value = cfg.smt_max_tokens || 512;
      DOM.cfgSmtDevice.value = cfg.smt_device || 'cuda';
      DOM.cfgOverwrite.checked = Boolean(cfg.overwrite);
      DOM.cfgSystemPrompt.value = cfg.system_prompt || '';

      DOM.lmstudioTestResult.textContent = '';
      DOM.modalSettings.style.display = 'flex';
    } catch (e) {
      alert('Ошибка чтения конфигурации: ' + e.message);
    }
  }

  function closeSettings() {
    DOM.modalSettings.style.display = 'none';
  }

  async function saveSettings() {
    const payload = {
      lm_host: DOM.cfgLmHost.value.trim(),
      lm_port: DOM.cfgLmPort.value.trim(),
      lm_model: DOM.cfgLmModel.value.trim(),
      lm_temperature: parseFloat(DOM.cfgLmTemp.value),
      lm_max_tokens: parseInt(DOM.cfgLmTokens.value, 10),
      qwen_context_length: parseInt(DOM.cfgQwenContext.value, 10),
      qwen_eval_batch_size: parseInt(DOM.cfgQwenBatch.value, 10),
      qwen_flash_attention: DOM.cfgFlashAtt.checked,
      dpi: parseInt(DOM.cfgDpi.value, 10),
      smt_max_tokens: parseInt(DOM.cfgSmtTokens.value, 10),
      smt_device: DOM.cfgSmtDevice.value,
      overwrite: DOM.cfgOverwrite.checked,
      system_prompt: DOM.cfgSystemPrompt.value
    };

    try {
      await api('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      closeSettings();
    } catch (e) {
      alert('Ошибка сохранения: ' + e.message);
    }
  }

  async function testLMStudio() {
    DOM.lmstudioTestResult.textContent = 'Проверка связи с LM Studio...';
    DOM.lmstudioTestResult.style.color = 'var(--text-secondary)';

    try {
      const res = await api('/api/lmstudio/test');
      if (res.online) {
        DOM.lmstudioTestResult.style.color = 'var(--accent-emerald)';
        DOM.lmstudioTestResult.textContent = `Успешно! ${res.message}`;
      } else {
        DOM.lmstudioTestResult.style.color = 'var(--accent-rose)';
        DOM.lmstudioTestResult.textContent = res.message;
      }
    } catch (e) {
      DOM.lmstudioTestResult.style.color = 'var(--accent-rose)';
      DOM.lmstudioTestResult.textContent = 'Ошибка проверки: ' + e.message;
    }
  }

  // ========================================================================
  // Visual Inspection Workbench (4 Modes)
  // ========================================================================
  async function loadWorkbenchPages(preserveCurrent = true) {
    try {
      const data = await api('/api/pages');
      const newPages = data.pages || [];
      const prevSelectedId = (preserveCurrent && state.pages[state.currentPageIndex]) 
        ? state.pages[state.currentPageIndex].page_id 
        : (DOM.pageSelect ? DOM.pageSelect.value : null);

      state.pages = newPages;

      DOM.pageSelect.innerHTML = '';
      state.pages.forEach((p, idx) => {
        const opt = document.createElement('option');
        opt.value = p.page_id;
        opt.textContent = `${idx + 1}. ${p.title} (${p.crops_count} станов)`;
        DOM.pageSelect.appendChild(opt);
      });

      if (state.pages.length === 0) {
        DOM.pageCounter.textContent = '0 / 0';
        state.pageData = null;
        renderActiveWorkbenchMode();
        return;
      }

      let targetId = state.pages[0].page_id;
      if (prevSelectedId && state.pages.some(p => p.page_id === prevSelectedId)) {
        targetId = prevSelectedId;
      }

      await selectWorkbenchPage(targetId);
    } catch (err) {
      console.warn('Load pages error:', err);
    }
  }

  // Client-Side LRU Page Cache (stores up to 35 pages for instant 0ms switching)
  const pageCache = new Map();

  function prefetchAdjacentPages(currentIndex) {
    if (!state.pages || state.pages.length === 0) return;
    const toPrefetch = [];
    if (currentIndex + 1 < state.pages.length) {
      toPrefetch.push(state.pages[currentIndex + 1].page_id);
    }
    if (currentIndex - 1 >= 0) {
      toPrefetch.push(state.pages[currentIndex - 1].page_id);
    }

    toPrefetch.forEach(pid => {
      if (!pageCache.has(pid)) {
        setTimeout(async () => {
          try {
            const data = await api(`/api/page/${pid}`);
            pageCache.set(pid, data);
            // Preload critical images in browser memory cache
            if (data.mask_url) { const img = new Image(); img.decoding = 'async'; img.src = data.mask_url; }
            if (data.debug_url) { const img = new Image(); img.decoding = 'async'; img.src = data.debug_url; }
          } catch (_) {}
        }, 80);
      }
    });
  }

  async function selectWorkbenchPage(pageId) {
    const idx = state.pages.findIndex(p => p.page_id === pageId);
    if (idx !== -1) state.currentPageIndex = idx;

    DOM.pageSelect.value = pageId;
    DOM.pageCounter.textContent = `${state.currentPageIndex + 1} / ${state.pages.length}`;

    // 1. Instant LRU Cache Hit (0 ms transition)
    if (pageCache.has(pageId)) {
      state.pageData = pageCache.get(pageId);
      state.selectedCropIndex = 0;
      renderActiveWorkbenchMode();
      prefetchAdjacentPages(state.currentPageIndex);
      return;
    }

    // 2. Network Fetch with caching and prefetching
    try {
      const data = await api(`/api/page/${pageId}`);
      pageCache.set(pageId, data);
      if (pageCache.size > 35) {
        const oldestKey = pageCache.keys().next().value;
        pageCache.delete(oldestKey);
      }
      state.pageData = data;
      state.selectedCropIndex = 0;
      renderActiveWorkbenchMode();
      prefetchAdjacentPages(state.currentPageIndex);
    } catch (err) {
      console.warn('Failed to load page data:', err);
    }
  }

  function switchWorkbenchMode(modeNum) {
    state.activeMode = modeNum;

    DOM.modeBtns.forEach(btn => {
      const mode = parseInt(btn.dataset.mode, 10);
      btn.classList.toggle('active', mode === modeNum);
    });

    Object.keys(DOM.panels).forEach(key => {
      DOM.panels[key].classList.toggle('active', parseInt(key, 10) === modeNum);
    });

    renderActiveWorkbenchMode();
  }

  function renderActiveWorkbenchMode() {
    if (!state.pageData) return;
    switch (state.activeMode) {
      case 1: renderMode1(); break;
      case 2: renderMode2(); break;
      case 3: renderMode3(); break;
      case 4: renderMode4(); break;
    }
  }

  function renderMode1() {
    const d = state.pageData;
    const showYolo = state.showYoloDebugMode1;

    if (showYolo) {
      DOM.scanDebugImg.src = d.debug_url || d.original_url || '';
      if (DOM.mode1LeftTitle) DOM.mode1LeftTitle.textContent = `Разметка YOLO OLA v2.0 (${d.title})`;
      if (DOM.btnToggleScanMode) {
        DOM.btnToggleScanMode.textContent = 'Исходный скан';
        DOM.btnToggleScanMode.style.borderColor = 'var(--accent-primary)';
      }
    } else {
      DOM.scanDebugImg.src = d.original_url || d.debug_url || '';
      if (DOM.mode1LeftTitle) DOM.mode1LeftTitle.textContent = `Исходный скан (${d.title})`;
      if (DOM.btnToggleScanMode) {
        DOM.btnToggleScanMode.textContent = 'Разметка YOLO';
        DOM.btnToggleScanMode.style.borderColor = 'rgba(255,255,255,0.12)';
      }
    }

    DOM.scanMaskImg.src = d.mask_url || '';
    DOM.badgeStavesCount.textContent = `${d.crops.length} станов`;

    if (DOM.badgeSpreadInfo) {
      if (d.is_spread) {
        DOM.badgeSpreadInfo.style.display = 'inline-flex';
        DOM.badgeSpreadInfo.textContent = d.spread_side === 'left' ? 'Разворот: левая стр.' : 'Разворот: правая стр.';
        DOM.badgeSpreadInfo.className = 'badge badge-grand';
      } else {
        DOM.badgeSpreadInfo.style.display = 'inline-flex';
        DOM.badgeSpreadInfo.textContent = 'Одиночный лист';
        DOM.badgeSpreadInfo.className = 'badge badge-staff';
      }
    }
  }

  function renderMode2() {
    const crops = state.pageData.crops || [];
    DOM.stavesSubbar2.innerHTML = '';
    if (crops.length === 0) {
      DOM.normRawImg.src = '';
      DOM.normDeskewImg.src = '';
      DOM.normStaffId.textContent = 'Нет станов';
      return;
    }

    if (state.selectedCropIndex >= crops.length) state.selectedCropIndex = 0;

    crops.forEach((crop, idx) => {
      const chip = document.createElement('button');
      chip.className = `staff-chip ${idx === state.selectedCropIndex ? 'active' : ''}`;
      const bendTxt = crop.bend_delta ? ` ~${crop.bend_delta}px` : '';
      chip.innerHTML = `<span>#${crop.index} ${crop.class}</span> <span class="badge badge-grand">${crop.skew_angle > 0 ? '+' : ''}${crop.skew_angle}°${bendTxt}</span>`;
      chip.addEventListener('click', () => {
        state.selectedCropIndex = idx;
        renderMode2();
      });
      DOM.stavesSubbar2.appendChild(chip);
    });

    const activeCrop = crops[state.selectedCropIndex];
    DOM.normRawImg.src = activeCrop.raw_url;
    DOM.normDeskewImg.src = activeCrop.deskew_url;
    const bendInfo = activeCrop.bend_delta ? ` • Прогиб: ${activeCrop.bend_delta} px` : '';
    DOM.normAngleBadge.textContent = `Поворот: ${activeCrop.skew_angle > 0 ? '+' : ''}${activeCrop.skew_angle}°${bendInfo}`;
    if (DOM.normDeskewTitle) DOM.normDeskewTitle.textContent = 'Выпрямленный + Real-CUGAN 2x SR';
    DOM.normRawDim.textContent = `${activeCrop.width} × ${activeCrop.height} px`;
    DOM.normStaffId.textContent = `#${activeCrop.index} (${activeCrop.id})`;
    DOM.normStaffClass.textContent = activeCrop.class;
    const srW = activeCrop.sr_width || (activeCrop.width * 2);
    const srH = activeCrop.sr_height || (activeCrop.height * 2);
    DOM.normStaffRes.textContent = `${activeCrop.width}×${activeCrop.height} ➔ ${srW}×${srH} px (2x SR)`;
    if (activeCrop.bend_delta >= 0.5) {
      DOM.normStaffStatus.textContent = `Выпрямлен (${activeCrop.bend_delta} px) + Real-CUGAN 2x SR`;
    } else if (Math.abs(activeCrop.skew_angle) >= 0.25) {
      DOM.normStaffStatus.textContent = `Доворот (${activeCrop.skew_angle}°) + Real-CUGAN 2x SR`;
    } else {
      DOM.normStaffStatus.textContent = 'Горизонтально + Real-CUGAN 2x SR';
    }
  }

  function renderMode3() {
    const crops = state.pageData.crops || [];
    DOM.stavesSubbar3.innerHTML = '';
    if (crops.length === 0) {
      DOM.omrCropImg.src = '';
      DOM.omrCodeBox.textContent = 'На странице не обнаружено станов.';
      if (DOM.omrSheetCanvas) DOM.omrSheetCanvas.innerHTML = '';
      if (DOM.omrSheetDiagnostics) DOM.omrSheetDiagnostics.textContent = 'Нет станов';
      return;
    }

    if (state.selectedCropIndex >= crops.length) state.selectedCropIndex = 0;

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
    DOM.omrCodeBox.textContent = state.activeCodeTab === 'abc' ? activeCrop.abc : (activeCrop.kern || 'Нет kern данных');

    // Immediately render visual score staves
    renderVisualScore();
  }

  let currentTuneObject = null;
  let codeDebounceTimer = null;
  let isPlayingAudio = false;

  async function renderVisualScore() {
    if (!state.pageData || !state.pageData.crops || !DOM.omrSheetCanvas) return;
    const crop = state.pageData.crops[state.selectedCropIndex];
    if (!crop) return;

    const t0 = performance.now();
    const isAbcTab = state.activeCodeTab === 'abc';
    const textContent = DOM.omrCodeBox.textContent.trim();

    if (DOM.omrSheetEngineBadge) {
      DOM.omrSheetEngineBadge.textContent = isAbcTab ? 'abcjs (Direct3D SVG)' : 'Verovio C++ (Native)';
    }

    if (!textContent) {
      DOM.omrSheetCanvas.innerHTML = '<div style="color: var(--text-muted); font-size: 12px;">Нет нотных данных для отображения.</div>';
      if (DOM.omrSheetDiagnostics) DOM.omrSheetDiagnostics.textContent = 'Пусто';
      if (DOM.omrRenderTiming) DOM.omrRenderTiming.textContent = 'Рендер: 0 мс';
      currentTuneObject = null;
      return;
    }

    if (isAbcTab) {
      if (typeof ABCJS !== 'undefined') {
        try {
          const containerWidth = DOM.omrSheetCanvas.clientWidth || 600;
          const tunes = ABCJS.renderAbc(DOM.omrSheetCanvas, textContent, {
            responsive: 'resize',
            staffwidth: Math.max(300, containerWidth - 30),
            add_classes: true,
            selectTypes: ['note', 'rest'],
            format: {
              titlefont: 'sans-serif 12 bold',
              gchordfont: 'sans-serif 10 italic'
            }
          });

          const elapsed = performance.now() - t0;
          if (DOM.omrRenderTiming) DOM.omrRenderTiming.textContent = `Рендер: ${elapsed.toFixed(1)} мс`;

          currentTuneObject = (tunes && tunes[0]) ? tunes[0] : null;

          if (currentTuneObject && typeof currentTuneObject.getWarnings === 'function') {
            const warnings = currentTuneObject.getWarnings();
            if (warnings && warnings.length > 0) {
              DOM.omrSheetDiagnostics.textContent = `Синтаксис: исправлено ${warnings.length} замечаний`;
              DOM.omrSheetDiagnostics.style.color = 'var(--accent-amber)';
            } else {
              DOM.omrSheetDiagnostics.textContent = 'Синтаксис корректен';
              DOM.omrSheetDiagnostics.style.color = 'var(--accent-emerald)';
            }
          } else {
            DOM.omrSheetDiagnostics.textContent = 'Отображение активно';
            DOM.omrSheetDiagnostics.style.color = 'var(--accent-emerald)';
          }
        } catch (err) {
          DOM.omrSheetDiagnostics.textContent = `Ошибка abcjs: ${err.message}`;
          DOM.omrSheetDiagnostics.style.color = 'var(--accent-rose)';
        }
      } else {
        DOM.omrSheetCanvas.innerHTML = '<div style="color: var(--accent-rose); font-size: 12px;">Библиотека abcjs не загружена.</div>';
      }
    } else {
      // Humdrum **kern tab -> backend Verovio C++
      DOM.omrSheetDiagnostics.textContent = 'Запрос к C++ Verovio...';
      try {
        const res = await api('/api/render_score', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ format: 'humdrum', text: textContent })
        });
        if (res.success && res.svg) {
          DOM.omrSheetCanvas.innerHTML = res.svg;
          if (DOM.omrRenderTiming) DOM.omrRenderTiming.textContent = `Verovio: ${res.latency_ms} мс`;
          DOM.omrSheetDiagnostics.textContent = 'Humdrum отображен';
          DOM.omrSheetDiagnostics.style.color = 'var(--accent-emerald)';
        } else {
          DOM.omrSheetCanvas.innerHTML = `<div style="color: var(--accent-rose); font-size: 12px;">${res.error || 'Ошибка парсинга Humdrum'}</div>`;
          DOM.omrSheetDiagnostics.textContent = 'Ошибка Humdrum';
          DOM.omrSheetDiagnostics.style.color = 'var(--accent-rose)';
        }
      } catch (err) {
        DOM.omrSheetDiagnostics.textContent = `Сбой API: ${err.message}`;
        DOM.omrSheetDiagnostics.style.color = 'var(--accent-rose)';
      }
    }
  }

  async function playCurrentTuneAudio() {
    if (typeof ABCJS === 'undefined' || !ABCJS.synth || !ABCJS.synth.supportsAudio()) {
      alert('Синтез звука не поддерживается в данном веб-окружении.');
      return;
    }
    if (!currentTuneObject) {
      alert('Сначала дождитесь отрисовки или введите корректный ABC-код.');
      return;
    }

    if (isPlayingAudio) return;
    isPlayingAudio = true;
    DOM.btnSynthPlay.disabled = true;
    const btnSpan = DOM.btnSynthPlay.querySelector('span');
    if (btnSpan) btnSpan.textContent = 'Играет...';

    try {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      if (audioContext.state === 'suspended') {
        await audioContext.resume();
      }

      const synth = new ABCJS.synth.CreateSynth();
      await synth.init({
        audioContext: audioContext,
        visualObj: currentTuneObject,
        millisecondsPerMeasure: 1600
      });
      await synth.prime();
      await synth.start();

      const durationMs = Math.min(20000, Math.max(3000, (synth.duration || 4) * 1000));
      setTimeout(() => {
        isPlayingAudio = false;
        DOM.btnSynthPlay.disabled = false;
        if (btnSpan) btnSpan.textContent = 'Прослушать';
      }, durationMs);
    } catch (err) {
      console.warn('Audio synth playback error:', err);
      isPlayingAudio = false;
      DOM.btnSynthPlay.disabled = false;
      if (btnSpan) btnSpan.textContent = 'Прослушать';
    }
  }

  function updateFinalLeftScan() {
    const d = state.pageData;
    if (!d) return;

    if (DOM.btnFinalViewOrig) DOM.btnFinalViewOrig.classList.toggle('active', state.finalLeftView === 'orig');
    if (DOM.btnFinalViewMask) DOM.btnFinalViewMask.classList.toggle('active', state.finalLeftView === 'mask');
    if (DOM.btnFinalViewDebug) DOM.btnFinalViewDebug.classList.toggle('active', state.finalLeftView === 'debug');

    if (state.finalLeftView === 'mask') {
      DOM.finalScanImg.src = d.mask_url || d.original_url;
      if (DOM.finalLeftTitle) DOM.finalLeftTitle.textContent = 'Маскированная страница для VLM';
      if (DOM.finalLeftBadge) {
        DOM.finalLeftBadge.textContent = 'Заплатки <!-- MUSIC_STUB_ID -->';
        DOM.finalLeftBadge.className = 'badge badge-grand';
      }
    } else if (state.finalLeftView === 'debug') {
      DOM.finalScanImg.src = d.debug_url || d.original_url;
      if (DOM.finalLeftTitle) DOM.finalLeftTitle.textContent = 'Разметка YOLO OLA v2.0';
      if (DOM.finalLeftBadge) {
        DOM.finalLeftBadge.textContent = 'Детекция станов';
        DOM.finalLeftBadge.className = 'badge badge-staff';
      }
    } else {
      DOM.finalScanImg.src = d.original_url || d.debug_url;
      if (DOM.finalLeftTitle) DOM.finalLeftTitle.textContent = 'Оригинальный скан страницы';
      if (DOM.finalLeftBadge) {
        DOM.finalLeftBadge.textContent = 'Контрольный эталон';
        DOM.finalLeftBadge.className = 'badge badge-staff';
      }
    }
  }

  function renderMode4() {
    const d = state.pageData;
    updateFinalLeftScan();

    if (state.isRawMarkdownView) {
      DOM.bookRenderedContent.style.display = 'none';
      DOM.bookRawEditor.style.display = 'block';
      DOM.bookRawEditor.value = d.markdown;
      DOM.btnToggleMdRaw.textContent = 'Предпросмотр книги';
    } else {
      DOM.bookRawEditor.style.display = 'none';
      DOM.bookRenderedContent.style.display = 'block';
      DOM.btnToggleMdRaw.textContent = 'Исходный Markdown';

      if (typeof marked !== 'undefined') {
        DOM.bookRenderedContent.innerHTML = marked.parse(d.markdown);
        // Render embedded ABC sheet music in the book reader
        if (typeof ABCJS !== 'undefined') {
          const cardSheets = DOM.bookRenderedContent.querySelectorAll('.music-card-sheet');
          cardSheets.forEach(sheetEl => {
            const abcRaw = decodeURIComponent(sheetEl.dataset.abc || '');
            if (abcRaw) {
              try {
                ABCJS.renderAbc(sheetEl, abcRaw, {
                  responsive: 'resize',
                  add_classes: true
                });
              } catch (e) {
                console.warn('Book sheet render error:', e);
              }
            }
          });
        }
      } else {
        DOM.bookRenderedContent.innerHTML = `<pre>${d.markdown}</pre>`;
      }
    }
  }

  // Setup Marked renderer with styled music card and embedded sheet
  function setupMarkedRenderer() {
    if (typeof marked === 'undefined') return;
    const renderer = new marked.Renderer();
    const origCode = renderer.code.bind(renderer);
    let cardIdx = 0;

    renderer.code = function (code, lang) {
      if (lang === 'abc') {
        cardIdx++;
        const cardId = `book-score-${cardIdx}`;
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
              <span class="music-card-title">${title}</span>
              <div>
                <span class="badge badge-staff">Ключ: ${key}</span>
                <span class="badge badge-grand">Метр: ${meter}</span>
              </div>
            </div>
            <div class="music-card-sheet" id="${cardId}" data-abc="${encodeURIComponent(code)}"></div>
            <details class="music-card-details">
              <summary>Показать код ABC</summary>
              <pre class="music-card-code">${code}</pre>
            </details>
          </div>
        `;
      }
      return origCode(code, lang);
    };

    marked.setOptions({ renderer: renderer });
  }

  // Keyboard Shortcuts Engine
  function setupKeyboard() {
    window.addEventListener('keydown', (e) => {
      const tag = e.target.tagName ? e.target.tagName.toLowerCase() : '';
      if (tag === 'input' || tag === 'textarea' || e.target.isContentEditable) return;

      if (e.key === 'Escape') {
        closeSettings();
        return;
      }

      // If in Workbench
      if (state.activeWorkspace === 'workbench') {
        if (e.key === '1') { e.preventDefault(); switchWorkbenchMode(1); }
        else if (e.key === '2') { e.preventDefault(); switchWorkbenchMode(2); }
        else if (e.key === '3') { e.preventDefault(); switchWorkbenchMode(3); }
        else if (e.key === '4') { e.preventDefault(); switchWorkbenchMode(4); }
        else if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {
          e.preventDefault();
          if (state.currentPageIndex > 0) {
            state.currentPageIndex--;
            selectWorkbenchPage(state.pages[state.currentPageIndex].page_id);
          }
        }
        else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {
          e.preventDefault();
          if (state.currentPageIndex < state.pages.length - 1) {
            state.currentPageIndex++;
            selectWorkbenchPage(state.pages[state.currentPageIndex].page_id);
          }
        }
        else if (e.key === 'ArrowUp') {
          e.preventDefault();
          if (state.selectedCropIndex > 0) {
            state.selectedCropIndex--;
            if (state.activeMode === 2) renderMode2();
            if (state.activeMode === 3) renderMode3();
          }
        }
        else if (e.key === 'ArrowDown') {
          e.preventDefault();
          if (state.pageData && state.pageData.crops && state.selectedCropIndex < state.pageData.crops.length - 1) {
            state.selectedCropIndex++;
            if (state.activeMode === 2) renderMode2();
            if (state.activeMode === 3) renderMode3();
          }
        }
      }
    });
  }

  // Event Listeners Binding
  function setupEvents() {
    // Workspace Switchers
    DOM.btnWsDashboard.addEventListener('click', () => switchWorkspace('dashboard'));
    DOM.btnWsWorkbench.addEventListener('click', () => switchWorkspace('workbench'));

    // Pipeline Controls
    DOM.btnPipelineStart.addEventListener('click', async () => {
      await api('/api/pipeline/start', { method: 'POST' });
      await pollPipelineStatus();
    });

    DOM.btnPipelinePause.addEventListener('click', async () => {
      await api('/api/pipeline/pause', { method: 'POST' });
      await pollPipelineStatus();
    });

    DOM.btnPipelineStop.addEventListener('click', async () => {
      await api('/api/pipeline/stop', { method: 'POST' });
      await pollPipelineStatus();
    });

    DOM.btnQueueClear.addEventListener('click', async () => {
      await api('/api/queue/clear', { method: 'POST' });
      await refreshQueue();
    });

    // Settings Modal
    DOM.btnOpenSettings.addEventListener('click', openSettings);
    DOM.btnCloseSettings.addEventListener('click', closeSettings);
    DOM.btnCancelSettings.addEventListener('click', closeSettings);
    DOM.btnSaveSettings.addEventListener('click', saveSettings);
    DOM.btnTestLmstudio.addEventListener('click', testLMStudio);

    // Workbench Navigation
    DOM.btnPrevPage.addEventListener('click', () => {
      if (state.currentPageIndex > 0) {
        state.currentPageIndex--;
        selectWorkbenchPage(state.pages[state.currentPageIndex].page_id);
      }
    });

    if (DOM.btnRefreshPages) {
      DOM.btnRefreshPages.addEventListener('click', () => {
        pageCache.clear();
        loadWorkbenchPages(true);
      });
    }

    DOM.btnNextPage.addEventListener('click', () => {
      if (state.currentPageIndex < state.pages.length - 1) {
        state.currentPageIndex++;
        selectWorkbenchPage(state.pages[state.currentPageIndex].page_id);
      }
    });

    DOM.pageSelect.addEventListener('change', (e) => {
      selectWorkbenchPage(e.target.value);
    });

    DOM.modeBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        switchWorkbenchMode(parseInt(btn.dataset.mode, 10));
      });
    });

    if (DOM.btnToggleScanMode) {
      DOM.btnToggleScanMode.addEventListener('click', () => {
        state.showYoloDebugMode1 = !state.showYoloDebugMode1;
        renderMode1();
      });
    }

    DOM.toggleGuidelines.addEventListener('change', (e) => {
      DOM.guidelinesLayer.style.display = e.target.checked ? 'block' : 'none';
    });

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

    DOM.btnCopyCode.addEventListener('click', () => {
      navigator.clipboard.writeText(DOM.omrCodeBox.textContent).then(() => {
        DOM.omrCopyStatus.style.display = 'inline';
        setTimeout(() => { DOM.omrCopyStatus.style.display = 'none'; }, 2000);
      });
    });

    if (DOM.btnSynthPlay) {
      DOM.btnSynthPlay.addEventListener('click', playCurrentTuneAudio);
    }

    if (DOM.omrCodeBox) {
      DOM.omrCodeBox.addEventListener('input', () => {
        clearTimeout(codeDebounceTimer);
        codeDebounceTimer = setTimeout(() => {
          if (state.pageData && state.pageData.crops) {
            const crop = state.pageData.crops[state.selectedCropIndex];
            if (crop) {
              if (state.activeCodeTab === 'abc') {
                crop.abc = DOM.omrCodeBox.textContent;
              } else {
                crop.kern = DOM.omrCodeBox.textContent;
              }
            }
          }
          renderVisualScore();
        }, 40);
      });
    }

    if (DOM.omrSheetCanvas && typeof ResizeObserver !== 'undefined') {
      const scoreObserver = new ResizeObserver(() => {
        if (state.activeMode === 3) {
          renderVisualScore();
        }
      });
      scoreObserver.observe(DOM.omrSheetCanvas);
    }

    let isDarkScoreTheme = false;
    if (DOM.btnToggleScoreTheme) {
      DOM.btnToggleScoreTheme.addEventListener('click', () => {
        isDarkScoreTheme = !isDarkScoreTheme;
        if (DOM.omrSheetViewport) {
          DOM.omrSheetViewport.classList.toggle('theme-paper', !isDarkScoreTheme);
          DOM.omrSheetViewport.classList.toggle('theme-dark', isDarkScoreTheme);
        }
        DOM.btnToggleScoreTheme.querySelector('span').textContent = isDarkScoreTheme ? 'Бумага: Темная' : 'Бумага: Белая';
      });
    }

    DOM.btnRerunOmr.addEventListener('click', async () => {
      if (!state.pageData || !state.pageData.crops) return;
      const crop = state.pageData.crops[state.selectedCropIndex];
      if (!crop) return;

      DOM.btnRerunOmr.disabled = true;
      DOM.btnRerunOmr.textContent = 'Распознавание...';

      try {
        const res = await api('/api/transcribe_crop', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ page_id: state.pageData.page_id, crop_stem: crop.crop_stem })
        });
        if (res.status === 'success') {
          crop.abc = res.abc;
          crop.kern = res.kern;
          crop.model_used = res.model_used;
          pageCache.delete(state.pageData.page_id);
          renderMode3();
        }
      } catch (err) {
        alert('Ошибка OMR: ' + err.message);
      } finally {
        DOM.btnRerunOmr.disabled = false;
        DOM.btnRerunOmr.textContent = 'Перераспознать стан';
      }
    });

    if (DOM.btnFinalViewOrig) {
      DOM.btnFinalViewOrig.addEventListener('click', () => {
        state.finalLeftView = 'orig';
        updateFinalLeftScan();
      });
    }
    if (DOM.btnFinalViewMask) {
      DOM.btnFinalViewMask.addEventListener('click', () => {
        state.finalLeftView = 'mask';
        updateFinalLeftScan();
      });
    }
    if (DOM.btnFinalViewDebug) {
      DOM.btnFinalViewDebug.addEventListener('click', () => {
        state.finalLeftView = 'debug';
        updateFinalLeftScan();
      });
    }

    DOM.btnToggleMdRaw.addEventListener('click', () => {
      state.isRawMarkdownView = !state.isRawMarkdownView;
      if (!state.isRawMarkdownView) {
        state.pageData.markdown = DOM.bookRawEditor.value;
      }
      renderMode4();
    });

    DOM.btnSaveMd.addEventListener('click', async () => {
      if (!state.pageData) return;
      const content = state.isRawMarkdownView ? DOM.bookRawEditor.value : state.pageData.markdown;
      try {
        const res = await api('/api/save_markdown', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ page_id: state.pageData.page_id, markdown: content })
        });
        if (res.status === 'success') {
          pageCache.delete(state.pageData.page_id);
          DOM.btnSaveMd.textContent = 'Сохранено';
          setTimeout(() => { DOM.btnSaveMd.textContent = 'Сохранить'; }, 2000);
        }
      } catch (e) {
        alert('Ошибка сохранения: ' + e.message);
      }
    });
  }

  // App Initialization
  async function init() {
    setupMarkedRenderer();
    setupDropzone();
    setupKeyboard();
    setupEvents();

    await refreshQueue();
    await loadWorkbenchPages();

    // Start Real-Time HUD Polling
    pollPipelineStatus();
    state.statusPollTimer = setInterval(pollPipelineStatus, 1500);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
