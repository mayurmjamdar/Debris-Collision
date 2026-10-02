// =========================================================
// CONFIGURATION
// =========================================================

const API_BASE = "http://127.0.0.1:8000";


// =========================================================
// ELEMENTS
// =========================================================

const csvFile = document.getElementById("csvFile");

const dropZone = document.getElementById("dropZone");

const fileName = document.getElementById("fileName");

const uploadResult =
    document.getElementById("uploadResult");

const eventSection =
    document.getElementById("eventSection");

const eventSelect =
    document.getElementById("eventSelect");

const eventCount =
    document.getElementById("eventCount");

const loadEventBtn =
    document.getElementById("loadEventBtn");

const dashboard =
    document.getElementById("dashboard");

const loading =
    document.getElementById("loading");

const toast =
    document.getElementById("toast");

const statusDot =
    document.getElementById("statusDot");

const statusText =
    document.getElementById("statusText");


// =========================================================
// API STATUS
// =========================================================

async function checkAPI() {

    try {

        const response =
            await fetch(
                `${API_BASE}/api/health`
            );

        if (!response.ok) {

            throw new Error(
                "API unavailable"
            );
        }


        statusDot.className =
            "status-dot online";

        statusText.textContent =
            "API Online";

    }

    catch (error) {

        statusDot.className =
            "status-dot offline";

        statusText.textContent =
            "API Offline";
    }
}


checkAPI();


// =========================================================
// FILE INPUT
// =========================================================

csvFile.addEventListener(
    "change",
    () => {

        if (
            csvFile.files &&
            csvFile.files.length > 0
        ) {

            handleFile(
                csvFile.files[0]
            );
        }
    }
);


// =========================================================
// DRAG & DROP
// =========================================================

dropZone.addEventListener(
    "dragover",
    (event) => {

        event.preventDefault();

        dropZone.classList.add(
            "dragging"
        );
    }
);


dropZone.addEventListener(
    "dragleave",
    () => {

        dropZone.classList.remove(
            "dragging"
        );
    }
);


dropZone.addEventListener(
    "drop",
    (event) => {

        event.preventDefault();

        dropZone.classList.remove(
            "dragging"
        );


        const files =
            event.dataTransfer.files;


        if (
            files.length > 0
        ) {

            handleFile(
                files[0]
            );
        }
    }
);


// =========================================================
// HANDLE FILE
// =========================================================

async function handleFile(file) {

    if (
        !file.name
        .toLowerCase()
        .endsWith(".csv")
    ) {

        showToast(
            "Please select a CSV file."
        );

        return;
    }


    fileName.textContent =
        `Selected: ${file.name}`;


    await uploadFile(file);
}


// =========================================================
// UPLOAD
// =========================================================

async function uploadFile(file) {

    showLoading(true);


    const formData =
        new FormData();


    formData.append(
        "file",
        file
    );


    try {

        const response =
            await fetch(
                `${API_BASE}/api/upload`,
                {
                    method: "POST",
                    body: formData
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Upload failed."
            );
        }


        uploadResult.classList.remove(
            "hidden"
        );


        uploadResult.innerHTML = `
            <strong>✓ Dataset uploaded</strong>
            <br>
            ${formatNumber(data.rows)}
            rows •
            ${formatNumber(data.columns)}
            columns •
            ${formatNumber(data.events)}
            events
        `;


        await loadEvents();


        eventSection.classList.remove(
            "hidden"
        );


        showToast(
            "Dataset uploaded successfully."
        );

    }

    catch (error) {

        showToast(
            error.message
        );

    }

    finally {

        showLoading(false);
    }
}


// =========================================================
// LOAD EVENTS
// =========================================================

async function loadEvents() {

    try {

        const response =
            await fetch(
                `${API_BASE}/api/events`
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Could not load events."
            );
        }


        eventSelect.innerHTML =
            `
            <option value="">
                Select an event...
            </option>
            `;


        data.events.forEach(
            (eventId) => {

                const option =
                    document.createElement(
                        "option"
                    );


                option.value =
                    eventId;


                option.textContent =
                    eventId;


                eventSelect.appendChild(
                    option
                );
            }
        );


        eventCount.textContent =
            `${formatNumber(data.count)} events`;

    }

    catch (error) {

        showToast(
            error.message
        );
    }
}


// =========================================================
// EVENT SELECTION
// =========================================================

eventSelect.addEventListener(
    "change",
    () => {

        loadEventBtn.disabled =
            eventSelect.value === "";
    }
);


loadEventBtn.addEventListener(
    "click",
    async () => {

        const eventId =
            eventSelect.value;


        if (!eventId) {

            return;
        }


        await loadEvent(eventId);
    }
);


// =========================================================
// LOAD EVENT
// =========================================================

async function loadEvent(eventId) {

    showLoading(true);

    dashboard.classList.add(
        "hidden"
    );


    try {

        const response =
            await fetch(
                `${API_BASE}/api/event/${encodeURIComponent(eventId)}`
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Could not load event."
            );
        }


        renderEvent(data);


        await runPrediction(
            eventId
        );


        dashboard.classList.remove(
            "hidden"
        );


        dashboard.scrollIntoView({
            behavior: "smooth"
        });

    }

    catch (error) {

        showToast(
            error.message
        );

    }

    finally {

        showLoading(false);
    }
}


// =========================================================
// RENDER EVENT
// =========================================================

function renderEvent(data) {

    document.getElementById(
        "eventIdDisplay"
    ).textContent =
        data.event_id;


    document.getElementById(
        "cdmCount"
    ).textContent =
        data.number_of_cdms;


    renderObject(
        "targetInfo",
        data.objects.target
    );


    renderObject(
        "chaserInfo",
        data.objects.chaser
    );


    const close =
        data.close_approach;


    setText(
        "timeToTca",
        formatNumber(
            close.time_to_tca,
            4
        )
    );


    setText(
        "missDistance",
        formatNumber(
            close.miss_distance,
            2
        )
    );


    setText(
        "relativeSpeed",
        formatNumber(
            close.relative_speed,
            2
        )
    );


    renderVector(
        "positionInfo",
        close.relative_position,
        "m"
    );


    renderVector(
        "velocityInfo",
        close.relative_velocity,
        "m/s"
    );


    renderTimeline(
        data.timeline
    );
}


// =========================================================
// OBJECT INFO
// =========================================================

function renderObject(
    elementId,
    object
) {

    const container =
        document.getElementById(
            elementId
        );


    const fields = [

        [
            "Semi-major axis",
            object.semi_major_axis
        ],

        [
            "Eccentricity",
            object.eccentricity
        ],

        [
            "Inclination",
            object.inclination
        ],

        [
            "Apogee",
            object.apogee
        ],

        [
            "Perigee",
            object.perigee
        ],

        [
            "RCS",
            object.rcs
        ]
    ];


    container.innerHTML = "";


    fields.forEach(
        ([label, value]) => {

            const div =
                document.createElement(
                    "div"
                );


            div.className =
                "info-item";


            div.innerHTML = `
                <span>${label}</span>
                <strong>
                    ${formatNumber(value)}
                </strong>
            `;


            container.appendChild(
                div
            );
        }
    );
}


// =========================================================
// VECTOR
// =========================================================

function renderVector(
    elementId,
    vector,
    unit
) {

    const container =
        document.getElementById(
            elementId
        );


    container.innerHTML = "";


    const axes = [

        ["R", vector?.r],

        ["T", vector?.t],

        ["N", vector?.n]

    ];


    axes.forEach(
        ([axis, value]) => {

            const div =
                document.createElement(
                    "div"
                );


            div.className =
                "vector";


            div.innerHTML = `
                <span class="axis">
                    ${axis}
                </span>

                <strong>
                    ${formatNumber(value)}
                    ${value !== null &&
                    value !== undefined
                        ? ` ${unit}`
                        : ""}
                </strong>
            `;


            container.appendChild(
                div
            );
        }
    );
}


// =========================================================
// PREDICTION
// =========================================================

async function runPrediction(
    eventId
) {

    try {

        const response =
            await fetch(
                `${API_BASE}/api/predict/${encodeURIComponent(eventId)}`,
                {
                    method: "POST"
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Prediction failed."
            );
        }


        if (
            data.prediction_available === false
        ) {

            setText(
                "predictedRisk",
                "N/A"
            );

            setText(
                "riskProbability",
                "N/A"
            );

            setText(
                "classification",
                "N/A"
            );


            document.getElementById(
                "predictionInfo"
            ).textContent =
                data.message;


            return;
        }


        setText(
            "predictedRisk",
            formatNumber(
                data.predicted_risk,
                4
            )
        );


        setText(
            "riskProbability",
            `${(
                data.high_risk_probability
                * 100
            ).toFixed(2)}%`
        );


        setText(
            "classification",
            data.classification
        );


        const box =
            document.getElementById(
                "classificationBox"
            );


        box.classList.remove(
            "high",
            "low"
        );


        if (
            data.classification === "HIGH"
        ) {

            box.classList.add(
                "high"
            );

        }

        else {

            box.classList.add(
                "low"
            );
        }


        document.getElementById(
            "predictionInfo"
        ).textContent =
            `Prediction generated using the latest `
            + `available CDM at `
            + `${formatNumber(
                data.prediction_cdm_time_to_tca,
                4
            )} days before TCA. `
            + `Classification threshold: `
            + `${data.classification_threshold}.`;

    }

    catch (error) {

        document.getElementById(
            "predictionInfo"
        ).textContent =
            `Prediction error: ${error.message}`;
    }
}


// =========================================================
// TIMELINE
// =========================================================

function renderTimeline(
    timeline
) {

    const container =
        document.getElementById(
            "timeline"
        );


    container.innerHTML = "";


    if (
        !timeline ||
        timeline.length === 0
    ) {

        container.innerHTML =
            `<p class="prediction-note">
                No timeline data available.
             </p>`;

        return;
    }


    const header =
        document.createElement(
            "div"
        );


    header.className =
        "timeline-row header";


    header.innerHTML = `
        <span>Time to TCA</span>
        <span>Miss Distance</span>
        <span>Relative Speed</span>
        <span>Risk</span>
    `;


    container.appendChild(
        header
    );


    timeline.forEach(
        (row) => {

            const div =
                document.createElement(
                    "div"
                );


            div.className =
                "timeline-row";


            div.innerHTML = `
                <span>
                    ${formatNumber(
                        row.time_to_tca,
                        4
                    )}
                </span>

                <span>
                    ${formatNumber(
                        row.miss_distance,
                        2
                    )}
                </span>

                <span>
                    ${formatNumber(
                        row.relative_speed,
                        2
                    )}
                </span>

                <span>
                    ${formatNumber(
                        row.risk,
                        4
                    )}
                </span>
            `;


            container.appendChild(
                div
            );
        }
    );
}


// =========================================================
// HELPERS
// =========================================================

function setText(
    id,
    value
) {

    const element =
        document.getElementById(id);


    if (element) {

        element.textContent =
            value;
    }
}


function formatNumber(
    value,
    decimals = 4
) {

    if (
        value === null ||
        value === undefined ||
        Number.isNaN(
            Number(value)
        )
    ) {

        return "N/A";
    }


    const number =
        Number(value);


    if (!Number.isFinite(number)) {

        return "N/A";
    }


    return number.toLocaleString(
        undefined,
        {
            maximumFractionDigits:
                decimals
        }
    );
}


function showLoading(
    show
) {

    if (show) {

        loading.classList.remove(
            "hidden"
        );

    }

    else {

        loading.classList.add(
            "hidden"
        );
    }
}


function showToast(
    message
) {

    toast.textContent =
        message;


    toast.classList.add(
        "show"
    );


    setTimeout(
        () => {

            toast.classList.remove(
                "show"
            );

        },
        3000
    );
}