/**
 * Ymir Theming System - Tailwind Configuration
 * Colors reference CSS variables for runtime theme switching
 */

tailwind.config = {
    theme: {
        extend: {
            colors: {
                // Primary brand color - references theme CSS variables
                'ymir': {
                    50: 'var(--color-primary-50)',
                    100: 'var(--color-primary-100)',
                    200: 'var(--color-primary-200)',
                    300: 'var(--color-primary-300)',
                    400: 'var(--color-primary-400)',
                    500: 'var(--color-primary-500)',
                    600: 'var(--color-primary-600)',
                    700: 'var(--color-primary-700)',
                    800: 'var(--color-primary-800)',
                    900: 'var(--color-primary-900)',
                    950: 'var(--color-primary-950)',
                },
                // Secondary/accent color
                'brand': {
                    50: 'var(--color-secondary-50)',
                    100: 'var(--color-secondary-100)',
                    200: 'var(--color-secondary-200)',
                    300: 'var(--color-secondary-300)',
                    400: 'var(--color-secondary-400)',
                    500: 'var(--color-secondary-500)',
                    600: 'var(--color-secondary-600)',
                    700: 'var(--color-secondary-700)',
                    800: 'var(--color-secondary-800)',
                    900: 'var(--color-secondary-900)',
                },
                // Semantic status colors
                'success': {
                    50: 'var(--color-success-50)',
                    100: 'var(--color-success-100)',
                    400: 'var(--color-success-400)',
                    500: 'var(--color-success-500)',
                    600: 'var(--color-success-600)',
                },
                'warning': {
                    50: 'var(--color-warning-50)',
                    100: 'var(--color-warning-100)',
                    400: 'var(--color-warning-400)',
                    500: 'var(--color-warning-500)',
                    600: 'var(--color-warning-600)',
                },
                'error': {
                    50: 'var(--color-error-50)',
                    100: 'var(--color-error-100)',
                    400: 'var(--color-error-400)',
                    500: 'var(--color-error-500)',
                    600: 'var(--color-error-600)',
                },
                'info': {
                    50: 'var(--color-info-50)',
                    100: 'var(--color-info-100)',
                    400: 'var(--color-info-400)',
                    500: 'var(--color-info-500)',
                    600: 'var(--color-info-600)',
                },
                // Surface/background colors
                'surface': {
                    'main': 'var(--color-bg-main)',
                    'sidebar': 'var(--color-bg-sidebar)',
                    'card': 'var(--color-bg-card)',
                    'input': 'var(--color-bg-input)',
                    'hover': 'var(--color-bg-hover)',
                },
                // Border colors
                'theme-border': {
                    DEFAULT: 'var(--color-border)',
                    'subtle': 'var(--color-border-subtle)',
                    'hover': 'var(--color-border-hover)',
                },
            },
            textColor: {
                'theme': {
                    'primary': 'var(--color-text-primary)',
                    'secondary': 'var(--color-text-secondary)',
                    'muted': 'var(--color-text-muted)',
                }
            },
            backgroundColor: {
                'surface': {
                    'main': 'var(--color-bg-main)',
                    'sidebar': 'var(--color-bg-sidebar)',
                    'card': 'var(--color-bg-card)',
                    'input': 'var(--color-bg-input)',
                    'hover': 'var(--color-bg-hover)',
                },
            },
            borderColor: {
                'theme': {
                    DEFAULT: 'var(--color-border)',
                    'subtle': 'var(--color-border-subtle)',
                    'hover': 'var(--color-border-hover)',
                },
            },
            fontFamily: {
                sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'Noto Sans', 'sans-serif'],
            }
        }
    }
};
