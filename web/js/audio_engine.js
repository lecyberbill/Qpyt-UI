/**
 * Qpyt-UI Audio Engine
 * Lightweight manager for UI sound effects
 */
class AudioEngine {
    constructor() {
        this.enabled = false;
        this.sounds = {
            start: 'https://cdn.pixabay.com/audio/2022/03/10/audio_c8c8a7345b.mp3', // Simple click/pop
            finish: 'https://cdn.pixabay.com/audio/2021/08/04/audio_0625c1539c.mp3', // Success chime
            error: 'https://cdn.pixabay.com/audio/2022/03/10/audio_783d4a023c.mp3'   // Soft alert
        };
        this.cache = {};
        this.loadSettings();
    }

    loadSettings() {
        const saved = localStorage.getItem('qpyt_audio_enabled');
        this.enabled = saved === 'true';
    }

    toggle(state) {
        this.enabled = state !== undefined ? state : !this.enabled;
        localStorage.setItem('qpyt_audio_enabled', this.enabled);
    }

    play(soundName) {
        if (!this.enabled || !this.sounds[soundName]) return;

        if (!this.cache[soundName]) {
            this.cache[soundName] = new Audio(this.sounds[soundName]);
        }

        const audio = this.cache[soundName];
        audio.currentTime = 0;
        audio.volume = 0.4;
        audio.play().catch(e => console.warn("[AudioEngine] Playback failed:", e));
    }
}

window.qpyt_audio = new AudioEngine();
