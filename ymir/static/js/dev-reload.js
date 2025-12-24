/**
 * Dev auto-reload - refreshes browser when server restarts
 * Only runs on localhost, safe to include in production builds
 */
(function() {
    // Only run on localhost
    if (!window.location.hostname.match(/localhost|127\.0\.0\.1/)) return;

    let reconnectInterval;

    function connect() {
        const ws = new WebSocket(`ws://${window.location.host}/dev/reload`);

        ws.onopen = () => {
            clearInterval(reconnectInterval);
        };

        ws.onclose = () => {
            // Server disconnected, poll until it's back then reload
            reconnectInterval = setInterval(() => {
                fetch('/')
                    .then(() => location.reload())
                    .catch(() => {}); // Server still down, keep polling
            }, 500);
        };
    }

    connect();
})();
