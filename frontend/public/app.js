/**
 * ClienteWeb: Controlador SPA para captura WebRTC, extracción transitoria en <canvas>
 * y consumo asíncrono de la API de Reconocimiento Facial a través del Reverse-Proxy HTTPS.
 *
 * Implementa la especificación del Diagrama de Clases en 'Propuesta Inicial.md':
 * - iniciarCamara()
 * - capturarFotograma()
 * - enviarInferencia()
 * - actualizarInterfaz(datos)
 */

class ClienteWeb {
  constructor() {
    this.apiBase = "/api";
    this.stream = null;
    this.facingMode = "user"; // 'user' (frontal) | 'environment' (trasera)
    this.autoScanActive = false;
    this.autoScanTimer = null;
    this.inferenceInFlight = false;
    this.enrollSamples = []; // Array<{ blob: Blob, previewUrl: string, name: string }>

    this._cacheDom();
    this._bindEvents();
    this._verificarContextoSeguro();
    this.consultarEstadoApi();
  }

  _cacheDom() {
    this.video = document.getElementById("videoStream");
    this.canvas = document.getElementById("captureCanvas");
    this.placeholder = document.getElementById("videoPlaceholder");
    this.reticle = document.getElementById("scanReticle");
    this.cameraBadge = document.getElementById("cameraBadge");

    this.pillHttps = document.getElementById("pillHttps");
    this.pillApi = document.getElementById("pillApi");
    this.securityAlert = document.getElementById("securityAlert");
    this.latencyBadge = document.getElementById("latencyBadge");

    this.btnToggleCamera = document.getElementById("btnToggleCamera");
    this.btnSwitchCamera = document.getElementById("btnSwitchCamera");
    this.btnIdentifyNow = document.getElementById("btnIdentifyNow");
    this.btnAutoScan = document.getElementById("btnAutoScan");

    this.resultIdle = document.getElementById("resultIdle");
    this.resultNotFound = document.getElementById("resultNotFound");
    this.notFoundTitle = document.getElementById("notFoundTitle");
    this.notFoundDetail = document.getElementById("notFoundDetail");
    this.resultProfile = document.getElementById("resultProfile");

    this.profNombres = document.getElementById("profNombres");
    this.profEdad = document.getElementById("profEdad");
    this.profId = document.getElementById("profId");
    this.profCarrera = document.getElementById("profCarrera");
    this.profMusica = document.getElementById("profMusica");
    this.profDistancia = document.getElementById("profDistancia");
    this.profConfianza = document.getElementById("profConfianza");

    // Pestañas y Enrolamiento
    this.tabButtons = document.querySelectorAll(".tab-btn");
    this.views = document.querySelectorAll(".panel-view");
    this.enrollForm = document.getElementById("enrollForm");
    this.btnCaptureSample = document.getElementById("btnCaptureSample");
    this.btnClearSamples = document.getElementById("btnClearSamples");
    this.inputFiles = document.getElementById("inputFiles");
    this.samplesGallery = document.getElementById("samplesGallery");
    this.samplesCountBadge = document.getElementById("samplesCountBadge");
    this.enrollFeedback = document.getElementById("enrollFeedback");
    this.btnSubmitEnroll = document.getElementById("btnSubmitEnroll");
  }

  _bindEvents() {
    this.btnToggleCamera.addEventListener("click", () => {
      if (this.stream) {
        this.detenerCamara();
      } else {
        this.iniciarCamara();
      }
    });

    this.btnSwitchCamera.addEventListener("click", async () => {
      this.facingMode = this.facingMode === "user" ? "environment" : "user";
      if (this.stream) {
        await this.iniciarCamara();
      }
    });

    this.btnIdentifyNow.addEventListener("click", () => {
      this.ejecutarCicloInferencia();
    });

    this.btnAutoScan.addEventListener("click", () => {
      if (this.autoScanActive) {
        this.detenerEscaneoContinuo();
      } else {
        this.iniciarEscaneoContinuo();
      }
    });

    this.tabButtons.forEach((btn) => {
      btn.addEventListener("click", () => this._cambiarVista(btn));
    });

    this.btnCaptureSample.addEventListener("click", async () => {
      try {
        const blob = await this.capturarFotograma();
        this._agregarMuestra(blob, `camara_${Date.now()}.jpg`);
      } catch (err) {
        this._mostrarFeedbackRegistro(err.message || "Error al capturar muestra", true);
      }
    });

    this.inputFiles.addEventListener("change", (ev) => {
      const files = Array.from(ev.target.files || []);
      files.forEach((file) => this._agregarMuestra(file, file.name));
      this.inputFiles.value = "";
    });

    this.btnClearSamples.addEventListener("click", () => this._limpiarMuestras());

    this.enrollForm.addEventListener("submit", (ev) => {
      ev.preventDefault();
      this.registrarNuevoUsuario();
    });
  }

  _verificarContextoSeguro() {
    if (window.isSecureContext) {
      this.pillHttps.textContent = "TLS / Contexto Seguro Activo";
      this.pillHttps.className = "pill pill-ok";
      this.securityAlert.classList.add("hidden");
    } else {
      this.pillHttps.textContent = "HTTP Inseguro (WebRTC restringido)";
      this.pillHttps.className = "pill pill-warn";
      this.securityAlert.classList.remove("hidden");
    }
  }

  async consultarEstadoApi() {
    try {
      const resp = await fetch(`${this.apiBase}/health`, { cache: "no-store" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      this.pillApi.textContent = `API Activa (${data.usuarios_registrados} usuarios / ${data.vectores_biometricos} vectores)`;
      this.pillApi.className = "pill pill-ok";
    } catch (_err) {
      this.pillApi.textContent = "API Sin Conexión";
      this.pillApi.className = "pill pill-error";
    }
  }

  _cambiarVista(clickedBtn) {
    const targetId = clickedBtn.dataset.tab;
    this.tabButtons.forEach((btn) => {
      const isActive = btn === clickedBtn;
      btn.classList.toggle("active", isActive);
      btn.setAttribute("aria-selected", String(isActive));
    });
    this.views.forEach((view) => {
      view.classList.toggle("hidden", view.id !== targetId);
    });
    if (targetId !== "viewInferencia" && this.autoScanActive) {
      this.detenerEscaneoContinuo();
    }
  }

  /**
   * Solicita acceso al hardware de video mediante WebRTC (navigator.mediaDevices.getUserMedia).
   */
  async iniciarCamara() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      this.actualizarInterfaz({
        reconocido: false,
        titulo: "WebRTC no disponible",
        detalle: "Asegúrate de acceder mediante HTTPS:// en la red local para habilitar permisos de cámara.",
      });
      return;
    }

    this.detenerCamara(false);

    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: this.facingMode },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    };

    try {
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.video.srcObject = this.stream;
      this.video.classList.toggle("mirrored", this.facingMode === "user");
      await this.video.play();

      this.placeholder.classList.add("hidden");
      this.cameraBadge.textContent = this.facingMode === "user" ? "En vivo (Frontal)" : "En vivo (Trasera)";
      this.cameraBadge.className = "badge badge-live";
      this.btnToggleCamera.textContent = "Detener Cámara";
      this.btnSwitchCamera.disabled = false;
      this.btnIdentifyNow.disabled = false;
      this.btnAutoScan.disabled = false;
      this.btnCaptureSample.disabled = false;
    } catch (err) {
      this.cameraBadge.textContent = "Permiso denegado / Error";
      this.cameraBadge.className = "badge pill-error";
      this.actualizarInterfaz({
        reconocido: false,
        titulo: "No fue posible acceder a la cámara",
        detalle: err.message || "Verifica los permisos de cámara del navegador y el certificado HTTPS.",
      });
    }
  }

  detenerCamara(resetUi = true) {
    if (this.autoScanActive) {
      this.detenerEscaneoContinuo();
    }
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    this.video.srcObject = null;

    if (resetUi) {
      this.placeholder.classList.remove("hidden");
      this.cameraBadge.textContent = "Cámara inactiva";
      this.cameraBadge.className = "badge badge-idle";
      this.btnToggleCamera.textContent = "Iniciar Cámara";
      this.btnSwitchCamera.disabled = true;
      this.btnIdentifyNow.disabled = true;
      this.btnAutoScan.disabled = true;
      this.btnCaptureSample.disabled = true;
    }
  }

  /**
   * Extrae un fotograma puntual del flujo de video en un Blob JPEG en memoria.
   * @returns {Promise<Blob>}
   */
  capturarFotograma() {
    return new Promise((resolve, reject) => {
      if (!this.stream || !this.video.videoWidth || !this.video.videoHeight) {
        reject(new Error("El flujo de video aún no está listo."));
        return;
      }

      this.canvas.width = this.video.videoWidth;
      this.canvas.height = this.video.videoHeight;
      const ctx = this.canvas.getContext("2d");
      ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

      this.canvas.toBlob(
        (blob) => {
          if (!blob) {
            reject(new Error("Error al codificar el fotograma en memoria."));
            return;
          }
          resolve(blob);
        },
        "image/jpeg",
        0.88
      );
    });
  }

  /**
   * Envía el fotograma capturado de forma asíncrona a POST /api/inferencia.
   * @param {Blob} fotogramaBlob
   * @returns {Promise<Object>}
   */
  async enviarInferencia(fotogramaBlob) {
    const formData = new FormData();
    formData.append("imagen", fotogramaBlob, "frame.jpg");

    const inicio = performance.now();
    const response = await fetch(`${this.apiBase}/inferencia`, {
      method: "POST",
      body: formData,
    });
    const latenciaMs = Math.round(performance.now() - inicio);
    this.latencyBadge.textContent = `Latencia: ${latenciaMs} ms`;
    this.latencyBadge.className = latenciaMs <= 3000 ? "badge badge-live" : "badge pill-warn";

    const payload = await response.json().catch(() => ({}));

    if (response.ok) {
      return { reconocido: true, perfil: payload };
    }
    return {
      reconocido: false,
      titulo: payload.error || "Usuario no reconocido",
      detalle: payload.detalle || "No se encontró coincidencia dentro del umbral de confianza (<= 0.60).",
    };
  }

  /**
   * Renderiza los datos biográficos recuperados o la alerta de no coincidencia.
   * @param {Object} resultado
   */
  actualizarInterfaz(resultado) {
    this.resultIdle.classList.add("hidden");

    if (resultado.reconocido && resultado.perfil) {
      const p = resultado.perfil;
      this.resultNotFound.classList.add("hidden");
      this.resultProfile.classList.remove("hidden");

      this.profNombres.textContent = p.nombres || "Sin especificar";
      this.profEdad.textContent = p.edad !== null && p.edad !== undefined ? `${p.edad} años` : "No registrada";
      this.profId.textContent = `#${p.id}`;
      this.profCarrera.textContent = p.carrera || "No especificada";
      this.profMusica.textContent = p.gustos_musicales || "No especificados";
      this.profDistancia.textContent = typeof p.distancia === "number" ? p.distancia.toFixed(4) : "--";
      this.profConfianza.textContent =
        typeof p.confianza === "number" ? `${(p.confianza * 100).toFixed(1)}%` : "--";
    } else {
      this.resultProfile.classList.add("hidden");
      this.resultNotFound.classList.remove("hidden");
      this.notFoundTitle.textContent = resultado.titulo || "Usuario no reconocido";
      this.notFoundDetail.textContent =
        resultado.detalle || "El rostro no fue detectado o no supera el umbral biométrico.";
    }
  }

  async ejecutarCicloInferencia() {
    if (this.inferenceInFlight || !this.stream) return;
    this.inferenceInFlight = true;
    this.reticle.classList.add("scanning");
    this.btnIdentifyNow.disabled = true;

    try {
      const blob = await this.capturarFotograma();
      const resultado = await this.enviarInferencia(blob);
      this.actualizarInterfaz(resultado);
    } catch (err) {
      this.actualizarInterfaz({
        reconocido: false,
        titulo: "Error de comunicación con el servidor",
        detalle: err.message || "Verifica la disponibilidad del contenedor backend.",
      });
    } finally {
      this.inferenceInFlight = false;
      this.reticle.classList.remove("scanning");
      if (this.stream) {
        this.btnIdentifyNow.disabled = false;
      }
    }
  }

  iniciarEscaneoContinuo() {
    if (!this.stream) return;
    this.autoScanActive = true;
    this.btnAutoScan.textContent = "Detener Escaneo Continuo";
    this.btnAutoScan.classList.add("active-scan");

    const loop = async () => {
      if (!this.autoScanActive) return;
      await this.ejecutarCicloInferencia();
      if (this.autoScanActive) {
        this.autoScanTimer = setTimeout(loop, 1400);
      }
    };
    loop();
  }

  detenerEscaneoContinuo() {
    this.autoScanActive = false;
    if (this.autoScanTimer) {
      clearTimeout(this.autoScanTimer);
      this.autoScanTimer = null;
    }
    this.btnAutoScan.textContent = "Iniciar Escaneo Continuo";
    this.btnAutoScan.classList.remove("active-scan");
  }

  // ==================== Gestión de Enrolamiento (1:N) ====================

  _agregarMuestra(blob, name) {
    const previewUrl = URL.createObjectURL(blob);
    this.enrollSamples.push({ blob, previewUrl, name });
    this._renderizarGaleriaMuestras();
  }

  _eliminarMuestra(index) {
    const [removed] = this.enrollSamples.splice(index, 1);
    if (removed) URL.revokeObjectURL(removed.previewUrl);
    this._renderizarGaleriaMuestras();
  }

  _limpiarMuestras() {
    this.enrollSamples.forEach((s) => URL.revokeObjectURL(s.previewUrl));
    this.enrollSamples = [];
    this._renderizarGaleriaMuestras();
  }

  _renderizarGaleriaMuestras() {
    this.samplesGallery.innerHTML = "";
    const total = this.enrollSamples.length;
    this.samplesCountBadge.textContent = `${total} ${total === 1 ? "muestra" : "muestras"}`;

    this.enrollSamples.forEach((sample, idx) => {
      const item = document.createElement("div");
      item.className = "sample-thumb";

      const img = document.createElement("img");
      img.src = sample.previewUrl;
      img.alt = `Muestra facial ${idx + 1}`;

      const btnRemove = document.createElement("button");
      btnRemove.type = "button";
      btnRemove.className = "sample-remove";
      btnRemove.textContent = "×";
      btnRemove.title = "Eliminar muestra";
      btnRemove.addEventListener("click", () => this._eliminarMuestra(idx));

      item.appendChild(img);
      item.appendChild(btnRemove);
      this.samplesGallery.appendChild(item);
    });
  }

  _mostrarFeedbackRegistro(mensaje, esError = false) {
    this.enrollFeedback.textContent = mensaje;
    this.enrollFeedback.className = `alert ${esError ? "alert-error" : "alert-success"}`;
    this.enrollFeedback.classList.remove("hidden");
  }

  async registrarNuevoUsuario() {
    const nombres = document.getElementById("inputNombres").value.trim();
    const edadRaw = document.getElementById("inputEdad").value.trim();
    const carrera = document.getElementById("inputCarrera").value.trim();
    const gustos = document.getElementById("inputMusica").value.trim();

    if (!nombres) {
      this._mostrarFeedbackRegistro("El campo 'Nombres Completos' es obligatorio.", true);
      return;
    }

    if (this.enrollSamples.length === 0) {
      this._mostrarFeedbackRegistro(
        "Debes capturar desde la cámara o subir al menos 1 fotografía con un rostro visible.",
        true
      );
      return;
    }

    const formData = new FormData();
    formData.append("nombres", nombres);
    if (edadRaw !== "") formData.append("edad", edadRaw);
    if (carrera !== "") formData.append("carrera", carrera);
    if (gustos !== "") formData.append("gustos_musicales", gustos);

    this.enrollSamples.forEach((sample, idx) => {
      formData.append("imagenes", sample.blob, sample.name || `muestra_${idx + 1}.jpg`);
    });

    this.btnSubmitEnroll.disabled = true;
    this.btnSubmitEnroll.textContent = "Procesando vectores 128D...";

    try {
      const resp = await fetch(`${this.apiBase}/registro`, {
        method: "POST",
        body: formData,
      });
      const data = await resp.json().catch(() => ({}));

      if (!resp.ok) {
        const msg = [data.error, data.detalle].filter(Boolean).join(" — ");
        this._mostrarFeedbackRegistro(msg || "No fue posible registrar el usuario.", true);
        return;
      }

      this._mostrarFeedbackRegistro(
        `Usuario '${data.usuario.nombres}' (ID #${data.usuario.id}) registrado con ${data.vectores_registrados} vector(es) biométrico(s).`,
        false
      );
      this.enrollForm.reset();
      this._limpiarMuestras();
      await this.consultarEstadoApi();
    } catch (err) {
      this._mostrarFeedbackRegistro(err.message || "Error de red durante el registro.", true);
    } finally {
      this.btnSubmitEnroll.disabled = false;
      this.btnSubmitEnroll.textContent = "Registrar Usuario y Vectores 128D";
    }
  }
}

window.addEventListener("DOMContentLoaded", () => {
  window.clienteFacial = new ClienteWeb();
});
