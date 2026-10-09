const MAX_EDGE = 1024;
const frame = (source, width, height, name, kind = "image", timestamp = null) => {
    const scale = Math.min(1, MAX_EDGE / Math.max(width, height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    canvas.getContext("2d").drawImage(source, 0, 0, canvas.width, canvas.height);
    return { id: crypto.randomUUID(), name, kind, timestamp, mime: "image/jpeg", data: canvas.toDataURL("image/jpeg", 0.88).split(",")[1] };
};
function eventOnce(target, event, action) {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => fail(), 20000);
        const cleanup = () => { clearTimeout(timer); target.removeEventListener(event, done); target.removeEventListener("error", fail); };
        const done = () => { cleanup(); resolve(); };
        const fail = () => { cleanup(); reject(new Error("Cannot read this media file. Try a browser-compatible MP4, image, or audio format.")); };
        target.addEventListener(event, done, { once: true });
        target.addEventListener("error", fail, { once: true });
        action();
    });
}
export async function prepareMedia(file, count = 4, start = 0, duration = 15) {
    if (file.type.startsWith("image/")) {
        const bitmap = await createImageBitmap(file);
        try { return [frame(bitmap, bitmap.width, bitmap.height, file.name)]; }
        finally { bitmap.close(); }
    }
    if (file.type.startsWith("video/")) {
        const video = document.createElement("video");
        const url = URL.createObjectURL(file);
        video.muted = true; video.preload = "metadata"; video.playsInline = true;
        try {
            await eventOnce(video, "loadeddata", () => { video.src = url; video.load(); });
            if (!Number.isFinite(video.duration) || !video.duration) throw new Error("Video duration is unavailable.");
            const refs = [];
            count = Math.max(1, Math.min(8, Number(count)));
            for (let i = 0; i < count; i++) {
                const time = (video.duration - 0.01) * (i + 0.5) / count;
                await eventOnce(video, "seeked", () => { video.currentTime = time; });
                refs.push(frame(video, video.videoWidth, video.videoHeight, `${file.name} · frame ${i + 1}`, "frame", time));
            }
            return refs;
        } finally { video.removeAttribute("src"); video.load(); URL.revokeObjectURL(url); }
    }
    if (file.type.startsWith("audio/") || /\.(wav|flac|mp3|m4a|ogg)$/i.test(file.name)) {
        if (file.size > 32 * 1024 * 1024) throw new Error("Use an audio file under 32 MiB; trim a short excerpt first.");
        const context = new AudioContext();
        try {
            const decoded = await context.decodeAudioData(await file.arrayBuffer());
            start = Math.max(0, Number(start));
            const seconds = Math.min(30, Number(duration), decoded.duration - start);
            if (!(seconds > 0)) throw new Error("Audio start is past the end of the file.");
            const rate = 24000, size = Math.floor(seconds * rate);
            const offline = new OfflineAudioContext(1, size, rate);
            const source = offline.createBufferSource(); source.buffer = decoded;
            source.connect(offline.destination); source.start(0, start, seconds);
            const samples = (await offline.startRendering()).getChannelData(0);
            const buffer = new ArrayBuffer(44 + samples.length * 2), view = new DataView(buffer);
            const str = (offset, text) => [...text].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
            str(0, "RIFF"); view.setUint32(4, buffer.byteLength - 8, true); str(8, "WAVE"); str(12, "fmt ");
            view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
            view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
            str(36, "data"); view.setUint32(40, samples.length * 2, true);
            samples.forEach((v, i) => view.setInt16(44 + i * 2, Math.round(Math.max(-1, Math.min(1, v)) * 32767), true));
            let binary = ""; const bytes = new Uint8Array(buffer);
            for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
            return [{ id: crypto.randomUUID(), name: `${file.name} · ${start.toFixed(1)}–${(start + seconds).toFixed(1)}s`, kind: "audio", mime: "audio/wav", data: btoa(binary) }];
        } finally { await context.close(); }
    }
    throw new Error("Choose an image, video or audio file supported by your browser.");
}
