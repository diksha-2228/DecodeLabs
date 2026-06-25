/* --- CleanFlow Frontend Logic --- */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const fileInfo = document.getElementById("file-info");
    const fileNameEl = document.getElementById("file-name");
    const fileSizeEl = document.getElementById("file-size");
    const removeFileBtn = document.getElementById("remove-file");
    const runBtn = document.getElementById("run-btn");
    const statusBadge = document.getElementById("status-badge");
    
    const terminalSection = document.getElementById("terminal-section");
    const spinner = document.getElementById("spinner");
    const consoleBody = document.getElementById("console-body");
    const errorAlert = document.getElementById("error-alert");
    const errorTitle = document.getElementById("error-title");
    const errorText = document.getElementById("error-text");
    
    // Result Sections
    const comparisonSection = document.getElementById("comparison-section");
    const visualsSection = document.getElementById("visuals-section");
    const previewSection = document.getElementById("preview-section");
    
    // Stats Elements
    const beforeRows = document.getElementById("before-rows");
    const beforeDim = document.getElementById("before-dim");
    const beforeDups = document.getElementById("before-dups");
    const beforeMissing = document.getElementById("before-missing");
    const beforeMissingList = document.getElementById("before-missing-list");
    
    const afterRows = document.getElementById("after-rows");
    const afterDim = document.getElementById("after-dim");
    const afterDups = document.getElementById("after-dups");
    const afterMissing = document.getElementById("after-missing");
    const droppedColsContainer = document.getElementById("dropped-cols-container");
    const addedColsContainer = document.getElementById("added-cols-container");
    
    // Charts
    const chartDistributions = document.getElementById("chart-distributions");
    const chartOutliers = document.getElementById("chart-outliers");
    const chartCorrelation = document.getElementById("chart-correlation");
    
    // Table Preview
    const tableHeaders = document.getElementById("table-headers");
    const tableBody = document.getElementById("table-body");
    
    // Modal Elements
    const imageModal = document.getElementById("image-modal");
    const modalClose = document.getElementById("modal-close");
    const modalImg = document.getElementById("modal-img");
    const modalCaption = document.getElementById("modal-caption");

    let selectedFile = null;
    let logPrintingTimeout = null;

    // --- File Ingestion handlers ---

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop zone on dragover
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    // Handle dropped files
    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // Handle file input click selection
    fileInput.addEventListener('change', (e) => {
        if (fileInput.files.length > 0) {
            handleFileSelect(fileInput.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.name.endsWith('.csv')) {
            showUploadError("Unsupported File", "Please upload a valid CSV file (.csv).");
            return;
        }

        selectedFile = file;
        fileNameEl.textContent = file.name;
        
        // Format size
        let sizeStr = "";
        if (file.size < 1024 * 1024) {
            sizeStr = `(${(file.size / 1024).toFixed(1)} KB)`;
        } else {
            sizeStr = `(${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
        }
        fileSizeEl.textContent = sizeStr;
        
        // Show file info card & update button
        fileInfo.classList.remove("hidden");
        runBtn.removeAttribute("disabled");
        errorAlert.classList.add("hidden");
    }

    // Remove selected file
    removeFileBtn.addEventListener("click", (e) => {
        e.preventDefault();
        resetFileInput();
    });

    function resetFileInput() {
        selectedFile = null;
        fileInput.value = "";
        fileInfo.classList.add("hidden");
        runBtn.setAttribute("disabled", "true");
        statusBadge.className = "status-badge idle";
        statusBadge.innerHTML = `<i class="fa-solid fa-circle-dot"></i> Pipeline Idle`;
    }

    function showUploadError(title, msg) {
        errorAlert.classList.remove("hidden");
        errorTitle.textContent = title;
        errorText.textContent = msg;
    }

    // --- Pipeline Run Orchestrator ---

    runBtn.addEventListener("click", () => {
        if (!selectedFile) return;

        // Reset UI sections from previous runs
        hideResultSections();
        clearTimeout(logPrintingTimeout);
        errorAlert.classList.add("hidden");
        
        // Update Status
        statusBadge.className = "status-badge running";
        statusBadge.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Running Pipeline`;
        runBtn.setAttribute("disabled", "true");
        removeFileBtn.setAttribute("disabled", "true");

        // Open Terminal
        terminalSection.classList.remove("hidden");
        spinner.classList.remove("hidden");
        consoleBody.innerHTML = `<div class="terminal-line command">> Uploading ${selectedFile.name} to server...</div>`;

        // Upload Phase
        const formData = new FormData();
        formData.append("file", selectedFile);

        fetch("/upload", {
            method: "POST",
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || "Upload failed") });
            }
            return response.json();
        })
        .then(data => {
            appendConsoleLine(`Uploaded file successfully saved to database.`, "success-msg");
            appendConsoleLine(`Initializing EDA & Cleaning pipeline modules...`, "command");
            
            // Execute Pipeline Phase
            return fetch("/run-pipeline", {
                method: "POST"
            });
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || "Pipeline execution failed") });
            }
            return response.json();
        })
        .then(results => {
            spinner.classList.add("hidden");
            appendConsoleLine(`Received processing metadata. Streaming logs...`, "system-msg");
            // Stream logs progressively
            streamLogs(results);
        })
        .catch(err => {
            // Error Handling
            spinner.classList.add("hidden");
            appendConsoleLine(`ERROR: ${err.message}`, "error-msg");
            
            statusBadge.className = "status-badge failed";
            statusBadge.innerHTML = `<i class="fa-solid fa-circle-xmark"></i> Pipeline Failed`;
            
            showUploadError("Pipeline Execution Failed", err.message);
            
            runBtn.removeAttribute("disabled");
            removeFileBtn.removeAttribute("disabled");
        });
    });

    function appendConsoleLine(text, className = "") {
        const line = document.createElement("div");
        line.className = `terminal-line ${className}`;
        line.textContent = text;
        consoleBody.appendChild(line);
        consoleBody.scrollTop = consoleBody.scrollHeight;
    }

    // Progressively print logs back to screen
    function streamLogs(results) {
        const logs = results.logs;
        let lineIndex = 0;

        function printNext() {
            if (lineIndex < logs.length) {
                const lineText = logs[lineIndex];
                let typeClass = "";
                
                // Classify text content
                if (lineText.startsWith("===") || lineText.startsWith("PHASE")) {
                    typeClass = "phase-title";
                } else if (lineText.includes("Error") || lineText.includes("FAILED")) {
                    typeClass = "error-msg";
                } else if (lineText.includes("DONE") || lineText.includes("PASSED") || lineText.includes("Saved final")) {
                    typeClass = "success-msg";
                }
                
                appendConsoleLine(lineText, typeClass);
                lineIndex++;
                
                // Faster typing rate for long files
                const delay = lineText.length === 0 ? 10 : 35;
                logPrintingTimeout = setTimeout(printNext, delay);
            } else {
                // Done printing logs
                statusBadge.className = "status-badge success";
                statusBadge.innerHTML = `<i class="fa-solid fa-circle-check"></i> Pipeline Completed`;
                runBtn.removeAttribute("disabled");
                removeFileBtn.removeAttribute("disabled");
                
                displayResults(results);
            }
        }
        printNext();
    }

    function hideResultSections() {
        comparisonSection.classList.add("hidden");
        visualsSection.classList.add("hidden");
        previewSection.classList.add("hidden");
    }

    // Render results from pipeline execution payload
    function displayResults(results) {
        // 1. Map Audit Metrics (Before)
        const before = results.before_stats;
        beforeRows.innerHTML = `${before.shape[0].toLocaleString()} <span class="subtext">rows</span>`;
        beforeDim.textContent = `${before.shape[1]} columns`;
        beforeDups.textContent = before.duplicate_count;
        
        // Calculate raw missing rate
        let totalMissing = 0;
        let totalCells = before.shape[0] * before.shape[1];
        before.missing_data.forEach(col => { totalMissing += col.count; });
        const missingPct = totalCells > 0 ? ((totalMissing / totalCells) * 100).toFixed(1) : 0;
        beforeMissing.textContent = `${missingPct}%`;

        // Render missing column details
        beforeMissingList.innerHTML = "";
        let hasMissing = false;
        before.missing_data.forEach(item => {
            if (item.count > 0) {
                hasMissing = true;
                const li = document.createElement("li");
                li.className = "missing-item";
                li.innerHTML = `
                    <span class="missing-item-col">${item.column}</span>
                    <div class="missing-item-bar-wrapper">
                        <div class="missing-item-bar">
                            <div class="missing-item-bar-fill" style="width: ${item.pct}%"></div>
                        </div>
                        <span class="missing-item-pct">${item.pct}%</span>
                    </div>
                `;
                beforeMissingList.appendChild(li);
            }
        });
        if (!hasMissing) {
            beforeMissingList.innerHTML = `<li class="missing-item-none"><i class="fa-solid fa-circle-check text-success"></i> No missing data in raw attributes.</li>`;
        }

        // 2. Map Audit Metrics (After)
        const after = results.after_stats;
        afterRows.innerHTML = `${after.shape[0].toLocaleString()} <span class="subtext">rows</span>`;
        afterDim.textContent = `${after.shape[1]} columns`;
        afterDups.textContent = after.duplicate_count;
        afterMissing.textContent = after.missing_count;

        // Map column additions/drops
        droppedColsContainer.innerHTML = "";
        if (after.columns_dropped.length > 0) {
            after.columns_dropped.forEach(col => {
                const pill = document.createElement("span");
                pill.className = "col-pill dropped";
                pill.textContent = col;
                droppedColsContainer.appendChild(pill);
            });
        } else {
            droppedColsContainer.innerHTML = `<span class="col-pill-none">No redundant columns dropped</span>`;
        }

        addedColsContainer.innerHTML = "";
        if (after.columns_added.length > 0) {
            after.columns_added.forEach(col => {
                const pill = document.createElement("span");
                pill.className = "col-pill added";
                pill.textContent = col;
                addedColsContainer.appendChild(pill);
            });
        } else {
            addedColsContainer.innerHTML = `<span class="col-pill-none">No engineering properties added</span>`;
        }

        // Show comparison cards
        comparisonSection.classList.remove("hidden");

        // 3. Map Visualizations
        if (results.charts.distributions) {
            chartDistributions.src = `data:image/png;base64,${results.charts.distributions}`;
        }
        if (results.charts.outliers) {
            chartOutliers.src = `data:image/png;base64,${results.charts.outliers}`;
        }
        if (results.charts.correlation) {
            chartCorrelation.src = `data:image/png;base64,${results.charts.correlation}`;
        }
        visualsSection.classList.remove("hidden");

        // 4. Map Cleaned Preview Table
        if (results.preview_rows.length > 0) {
            tableHeaders.innerHTML = "";
            tableBody.innerHTML = "";

            const cols = Object.keys(results.preview_rows[0]);
            const engineeredCols = ["price_per_sqft", "total_rooms", "is_new_construction", "proximity_score"];

            // Create Headers
            cols.forEach(col => {
                const th = document.createElement("th");
                th.textContent = col;
                if (engineeredCols.includes(col)) {
                    th.classList.add("col-engineered");
                }
                tableHeaders.appendChild(th);
            });

            // Create Rows
            results.preview_rows.forEach(row => {
                const tr = document.createElement("tr");
                cols.forEach(col => {
                    const td = document.createElement("td");
                    const val = row[col];
                    
                    if (val === null || val === undefined) {
                        td.textContent = "NaN";
                        td.style.color = "var(--color-text-light)";
                    } else if (typeof val === 'number') {
                        // Format floats vs ints
                        td.textContent = Number.isInteger(val) ? val : val.toFixed(2);
                    } else {
                        td.textContent = val;
                    }

                    if (engineeredCols.includes(col)) {
                        td.classList.add("col-engineered");
                    }
                    tr.appendChild(td);
                });
                tableBody.appendChild(tr);
            });

            previewSection.classList.remove("hidden");
        }
    }

    // --- Zoom Modal image gallery ---

    const images = [chartDistributions, chartOutliers, chartCorrelation];
    images.forEach(img => {
        img.addEventListener("click", () => {
            if (!img.src) return;
            
            modalImg.src = img.src;
            
            // Set caption text
            const title = img.closest(".visual-card").querySelector("h3").textContent;
            const caption = img.closest(".visual-card").querySelector(".visual-footer p").textContent;
            modalCaption.innerHTML = `<strong>${title}</strong> &mdash; ${caption}`;
            
            imageModal.classList.add("active");
        });
    });

    // Close Modal triggers
    modalClose.addEventListener("click", closeModal);
    imageModal.addEventListener("click", (e) => {
        if (e.target === imageModal || e.target.closest(".modal-close")) {
            closeModal();
        }
    });

    // Escape key binds
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && imageModal.classList.contains("active")) {
            closeModal();
        }
    });

    function closeModal() {
        imageModal.classList.remove("active");
        setTimeout(() => {
            modalImg.src = "";
            modalCaption.textContent = "";
        }, 200);
    }
});
