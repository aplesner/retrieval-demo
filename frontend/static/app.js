/**
 * Multimodal Retrieval Demo - Frontend JavaScript
 */

// Dynamically determine API base URL based on deployment mode
const API_BASE = window.location.port === '80' || window.location.port === ''
    ? `${window.location.protocol}//${window.location.host}` // Production: same host/port (nginx proxy)
    : window.location.hostname === 'localhost'
        ? 'http://localhost:8080' // Local dev: backend on 8080
        : `http://${window.location.hostname}:8080`; // Remote dev: backend on 8080

// State
let allImages = [];
let selectedImageId = null;
let isLoading = false;

// DOM Elements (set after DOMContentLoaded)
let searchInput, searchBtn, resultsGrid, resultsCount, suggestionsContainer;
let imageSelector, selectedImagePreview, weightSlider, weightValue;

/**
 * Initialize the application
 */
async function init() {
    // Get DOM elements
    searchInput = document.getElementById('search-input');
    searchBtn = document.getElementById('search-btn');
    resultsGrid = document.getElementById('results-grid');
    resultsCount = document.getElementById('results-count');
    suggestionsContainer = document.getElementById('suggestions');
    imageSelector = document.getElementById('image-selector');
    selectedImagePreview = document.getElementById('selected-image');
    weightSlider = document.getElementById('weight-slider');
    weightValue = document.getElementById('weight-value');

    // Setup event listeners
    if (searchInput) {
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') handleSearch();
        });
    }
    if (searchBtn) {
        searchBtn.addEventListener('click', handleSearch);
    }
    if (weightSlider) {
        weightSlider.addEventListener('input', (e) => {
            if (weightValue) {
                weightValue.textContent = `${Math.round(e.target.value * 100)}%`;
            }
        });
    }

    // Load initial data
    await checkHealth();
    await loadSuggestions();
    await loadAllImages();
}

/**
 * Check backend health
 */
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();
        updateStatus(true, `${data.indexed_images} images indexed`);
    } catch (err) {
        updateStatus(false, 'Backend unavailable');
        console.error('Health check failed:', err);
    }
}

/**
 * Load suggested queries
 */
async function loadSuggestions() {
    if (!suggestionsContainer) return;

    try {
        const res = await fetch(`${API_BASE}/api/suggestions`);
        const data = await res.json();

        // Determine which suggestions to show based on page mode
        const pageMode = document.body.dataset.mode || 'text';
        let suggestions = [];

        switch(pageMode) {
            case 'text':
            case 'image':
            case 'multimodal':
                suggestions = data.image_suggestions || [];
                break;
            case 'audio':
                suggestions = data.audio_suggestions || [];
                break;
            case 'pdf':
                suggestions = data.pdf_suggestions || [];
                break;
        }

        const chips = suggestions.map(s =>
            `<button class="suggestion-chip" onclick="selectSuggestion('${s}')">${s}</button>`
        ).join('');

        suggestionsContainer.innerHTML = `
            <div class="suggestions-title">Try searching for:</div>
            <div class="suggestion-chips">${chips}</div>
        `;
    } catch (err) {
        console.error('Failed to load suggestions:', err);
    }
}

/**
 * Load all images for gallery/selection
 */
async function loadAllImages() {
    try {
        const res = await fetch(`${API_BASE}/api/images?limit=100`);
        allImages = await res.json();
        
        // Show all images by default
        displayResults(allImages.map(img => ({...img, score: null})));
    } catch (err) {
        console.error('Failed to load images:', err);
        displayError('Failed to load images');
    }
}

/**
 * Handle search based on current page mode
 */
async function handleSearch() {
    const query = searchInput?.value?.trim();
    const pageMode = document.body.dataset.mode || 'text';
    
    if (pageMode === 'text' && !query) {
        return;
    }
    if (pageMode === 'image' && !selectedImageId) {
        alert('Please select an image first');
        return;
    }
    if (pageMode === 'multimodal' && !query && !selectedImageId) {
        return;
    }
    
    setLoading(true);
    
    try {
        let results;
        
        switch (pageMode) {
            case 'text':
                results = await searchByText(query);
                break;
            case 'image':
                results = await searchByImage(selectedImageId);
                break;
            case 'multimodal':
                const weight = weightSlider?.value || 0.5;
                results = await searchMultimodal(query, selectedImageId, 1 - parseFloat(weight));
                break;
        }
        
        displayResults(results);
    } catch (err) {
        console.error('Search failed:', err);
        displayError('Search failed. Please try again.');
    } finally {
        setLoading(false);
    }
}

/**
 * Search by text query
 */
async function searchByText(query) {
    const res = await fetch(`${API_BASE}/api/search/text`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query, limit: 50})
    });
    const data = await res.json();
    return data.results;
}

/**
 * Search by image
 */
async function searchByImage(imageId) {
    const res = await fetch(`${API_BASE}/api/search/image`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({image_id: imageId, limit: 50})
    });
    const data = await res.json();
    return data.results;
}

/**
 * Search with text + image
 */
async function searchMultimodal(textQuery, imageId, textWeight) {
    const res = await fetch(`${API_BASE}/api/search/multimodal`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            text_query: textQuery || '',
            image_id: imageId,
            text_weight: textWeight,
            limit: 50
        })
    });
    const data = await res.json();
    return data.results;
}

/**
 * Display search results
 */
function displayResults(results) {
    if (!resultsGrid) return;
    
    if (resultsCount) {
        resultsCount.textContent = `${results.length} results`;
    }
    
    if (results.length === 0) {
        resultsGrid.innerHTML = `
            <div class="empty-state">
                <p>No results found</p>
            </div>
        `;
        return;
    }
    
    resultsGrid.innerHTML = results.map(img => `
        <div class="image-card ${img.id === selectedImageId ? 'selected' : ''}" 
             onclick="selectImage('${img.id}', '${img.path}')"
             data-id="${img.id}">
            <img src="${API_BASE}${img.path}" alt="${img.filename}" loading="lazy">
            <div class="image-card-info">
                ${img.score !== null ? `<div class="image-card-score">Score: ${(img.score * 100).toFixed(1)}%</div>` : ''}
                <div class="image-card-name">${img.filename}</div>
            </div>
        </div>
    `).join('');
}

/**
 * Select an image (for image search or multimodal)
 */
function selectImage(imageId, imagePath) {
    const pageMode = document.body.dataset.mode || 'text';
    
    // In text mode, clicking shows similar images
    if (pageMode === 'text') {
        selectedImageId = imageId;
        // Optionally trigger image search
        return;
    }
    
    // Update selection
    selectedImageId = imageId;
    
    // Update visual selection
    document.querySelectorAll('.image-card').forEach(card => {
        card.classList.toggle('selected', card.dataset.id === imageId);
    });
    
    // Update image selector preview
    if (imageSelector && selectedImagePreview) {
        imageSelector.classList.add('has-image');
        selectedImagePreview.src = `${API_BASE}${imagePath}`;
        selectedImagePreview.style.display = 'block';
        imageSelector.querySelector('.image-selector-text').style.display = 'none';
    }
    
    // Auto-search for image-only mode
    if (pageMode === 'image') {
        handleSearch();
    }
}

/**
 * Select a suggestion
 */
function selectSuggestion(text) {
    if (searchInput) {
        searchInput.value = text;
        handleSearch();
    }
}

/**
 * Set loading state
 */
function setLoading(loading) {
    isLoading = loading;
    
    if (searchBtn) {
        searchBtn.disabled = loading;
        searchBtn.textContent = loading ? 'Searching...' : 'Search';
    }
    
    if (loading && resultsGrid) {
        resultsGrid.innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <p>Searching...</p>
            </div>
        `;
    }
}

/**
 * Display error message
 */
function displayError(message) {
    if (resultsGrid) {
        resultsGrid.innerHTML = `
            <div class="empty-state">
                <p style="color: #ef4444;">${message}</p>
            </div>
        `;
    }
}

/**
 * Update status indicator
 */
function updateStatus(online, message) {
    const statusEl = document.getElementById('status');
    if (statusEl) {
        statusEl.innerHTML = `
            <div class="status-dot" style="background: ${online ? '#10b981' : '#ef4444'}"></div>
            <span>${message}</span>
        `;
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', init);
