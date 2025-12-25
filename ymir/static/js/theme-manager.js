/**
 * Ymir Theme Manager
 * Handles theme switching, persistence, and events
 */

const YmirTheme = {
    STORAGE_KEY: 'ymir-theme',
    DEFAULT_THEME: 'survey-corps',

    availableThemes: {
        'survey-corps': {
            name: 'Survey Corps',
            description: 'Attack on Titan inspired theme'
        },
        'default': {
            name: 'Classic Red',
            description: 'Original Ymir theme - red and slate'
        }
    },

    /**
     * Initialize theme on page load
     */
    init() {
        const savedTheme = localStorage.getItem(this.STORAGE_KEY) || this.DEFAULT_THEME;
        this.setTheme(savedTheme, false);
    },

    /**
     * Set the active theme
     * @param {string} themeName - Theme identifier
     * @param {boolean} save - Whether to persist to localStorage
     */
    setTheme(themeName, save = true) {
        if (!this.availableThemes[themeName]) {
            console.warn(`Theme "${themeName}" not found, using default`);
            themeName = this.DEFAULT_THEME;
        }

        document.documentElement.setAttribute('data-theme', themeName);

        if (save) {
            localStorage.setItem(this.STORAGE_KEY, themeName);
        }

        // Dispatch event for components that need to react
        document.dispatchEvent(new CustomEvent('themeChanged', {
            detail: {
                theme: themeName,
                themeInfo: this.availableThemes[themeName]
            }
        }));

        console.log(`Theme set to: ${themeName}`);
    },

    /**
     * Get current theme name
     * @returns {string}
     */
    getTheme() {
        return document.documentElement.getAttribute('data-theme') || this.DEFAULT_THEME;
    },

    /**
     * Get theme info object
     * @param {string} themeName - Optional theme name, defaults to current
     * @returns {object}
     */
    getThemeInfo(themeName = null) {
        const theme = themeName || this.getTheme();
        return this.availableThemes[theme] || null;
    },

    /**
     * Toggle to next theme in list
     */
    toggle() {
        const themes = Object.keys(this.availableThemes);
        const currentIndex = themes.indexOf(this.getTheme());
        const nextIndex = (currentIndex + 1) % themes.length;
        this.setTheme(themes[nextIndex]);
    },

    /**
     * Get list of all available themes
     * @returns {Array<{id: string, name: string, description: string}>}
     */
    listThemes() {
        return Object.entries(this.availableThemes).map(([id, info]) => ({
            id,
            ...info,
            active: id === this.getTheme()
        }));
    }
};

// Initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => YmirTheme.init());
} else {
    YmirTheme.init();
}

// Global exposure
window.YmirTheme = YmirTheme;
