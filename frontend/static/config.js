/**
 * Shared configuration for all pages
 */

// Dynamically determine API base URL based on deployment mode
const API_BASE = window.location.port === '80' || window.location.port === ''
    ? `${window.location.protocol}//${window.location.host}` // Production: same host/port (nginx proxy)
    : window.location.hostname === 'localhost'
        ? 'http://localhost:8080' // Local dev: backend on 8080
        : `http://${window.location.hostname}:8080`; // Remote dev: backend on 8080
