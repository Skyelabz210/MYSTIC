// MYSTIC GitHub Pages - JavaScript

/**
 * Navigation Scroll Effect
 */
function initNavbarScroll() {
    const navbar = document.querySelector('.navbar');
    let lastScrollY = window.scrollY;

    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
        lastScrollY = window.scrollY;
    });
}

/**
 * Mobile Menu Toggle
 */
function initMobileMenu() {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');

    if (hamburger) {
        hamburger.addEventListener('click', () => {
            navMenu.classList.toggle('active');
            hamburger.classList.toggle('active');

            // Toggle hamburger animation
            const spans = hamburger.querySelectorAll('span');
            if (hamburger.classList.contains('active')) {
                spans[0].style.transform = 'rotate(45deg) translate(5px, 5px)';
                spans[1].style.opacity = '0';
                spans[2].style.transform = 'rotate(-45deg) translate(7px, -6px)';
            } else {
                spans[0].style.transform = 'none';
                spans[1].style.opacity = '1';
                spans[2].style.transform = 'none';
            }
        });

        // Close menu when clicking a link
        const navLinks = navMenu.querySelectorAll('a');
        navLinks.forEach(link => {
            link.addEventListener('click', () => {
                navMenu.classList.remove('active');
                hamburger.classList.remove('active');
                const spans = hamburger.querySelectorAll('span');
                spans[0].style.transform = 'none';
                spans[1].style.opacity = '1';
                spans[2].style.transform = 'none';
            });
        });
    }
}

/**
 * Smooth Scroll for Anchor Links
 */
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href !== '#') {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    const offsetTop = target.offsetTop - 80; // Offset for fixed navbar
                    window.scrollTo({
                        top: offsetTop,
                        behavior: 'smooth'
                    });
                }
            }
        });
    });
}

/**
 * Intersection Observer for Fade-In Animations
 */
function initScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe all cards and sections
    const animatedElements = document.querySelectorAll(
        '.innovation-card, .problem-card, .validation-card, .doc-card, .impact-card, .architecture-layer'
    );

    animatedElements.forEach(el => {
        observer.observe(el);
    });
}

/**
 * Innovation Card Interactions
 */
function initInnovationCards() {
    const cards = document.querySelectorAll('.innovation-card');

    cards.forEach(card => {
        card.addEventListener('mouseenter', () => {
            const innovationNum = card.dataset.innovation;
            card.style.transform = 'translateY(-10px) scale(1.02)';
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = '';
        });
    });
}

/**
 * Dynamic Stats Counter Animation
 */
function animateStats() {
    const stats = document.querySelectorAll('.stat-value');
    const observerOptions = {
        threshold: 0.5
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !entry.target.classList.contains('counted')) {
                const target = entry.target;
                const text = target.textContent;

                // Only animate if it's a number
                if (text.match(/^\d+/)) {
                    const finalValue = parseInt(text);
                    let currentValue = 0;
                    const increment = finalValue / 50;
                    const timer = setInterval(() => {
                        currentValue += increment;
                        if (currentValue >= finalValue) {
                            target.textContent = text; // Restore original text
                            clearInterval(timer);
                            target.classList.add('counted');
                        } else {
                            target.textContent = Math.floor(currentValue).toLocaleString();
                        }
                    }, 30);
                } else {
                    target.classList.add('counted');
                }

                observer.unobserve(target);
            }
        });
    }, observerOptions);

    stats.forEach(stat => observer.observe(stat));
}

/**
 * Add Active State to Navigation Links
 */
function updateActiveNavLink() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-menu a[href^="#"]');

    window.addEventListener('scroll', () => {
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop - 100;
            const sectionHeight = section.offsetHeight;
            if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });
}

/**
 * Copy Code to Clipboard
 */
function initCodeCopy() {
    const codeBlocks = document.querySelectorAll('pre code');

    codeBlocks.forEach(block => {
        const pre = block.parentElement;
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        button.style.cssText = `
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            padding: 0.5rem 1rem;
            background-color: rgba(255, 255, 255, 0.1);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s;
        `;

        pre.style.position = 'relative';
        pre.appendChild(button);

        button.addEventListener('click', async () => {
            try {
                await navigator.clipboard.writeText(block.textContent);
                button.textContent = 'Copied!';
                button.style.backgroundColor = 'rgba(16, 185, 129, 0.8)';

                setTimeout(() => {
                    button.textContent = 'Copy';
                    button.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
                }, 2000);
            } catch (err) {
                console.error('Failed to copy:', err);
            }
        });

        button.addEventListener('mouseenter', () => {
            button.style.backgroundColor = 'rgba(255, 255, 255, 0.2)';
        });

        button.addEventListener('mouseleave', () => {
            if (button.textContent === 'Copy') {
                button.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
            }
        });
    });
}

/**
 * Performance Monitoring (Optional)
 */
function logPerformance() {
    if (window.performance && window.performance.timing) {
        window.addEventListener('load', () => {
            setTimeout(() => {
                const perfData = window.performance.timing;
                const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;
                console.log(`Page Load Time: ${pageLoadTime}ms`);
            }, 0);
        });
    }
}

/**
 * Easter Egg: Konami Code
 */
function initKonamiCode() {
    const konamiCode = [38, 38, 40, 40, 37, 39, 37, 39, 66, 65]; // ↑↑↓↓←→←→BA
    let konamiIndex = 0;

    document.addEventListener('keydown', (e) => {
        if (e.keyCode === konamiCode[konamiIndex]) {
            konamiIndex++;
            if (konamiIndex === konamiCode.length) {
                activateEasterEgg();
                konamiIndex = 0;
            }
        } else {
            konamiIndex = 0;
        }
    });
}

function activateEasterEgg() {
    // Create floating φ symbols
    for (let i = 0; i < 50; i++) {
        setTimeout(() => {
            const phi = document.createElement('div');
            phi.textContent = 'φ';
            phi.style.cssText = `
                position: fixed;
                font-size: 2rem;
                font-weight: 700;
                background: linear-gradient(135deg, #0066ff, #6366f1);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                pointer-events: none;
                z-index: 9999;
                left: ${Math.random() * 100}vw;
                top: -50px;
                animation: fall ${3 + Math.random() * 2}s linear;
            `;

            document.body.appendChild(phi);

            setTimeout(() => phi.remove(), 5000);
        }, i * 100);
    }

    // Add fall animation
    if (!document.getElementById('phi-animation')) {
        const style = document.createElement('style');
        style.id = 'phi-animation';
        style.textContent = `
            @keyframes fall {
                to {
                    transform: translateY(100vh) rotate(360deg);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    }

    console.log('🎉 MYSTIC Easter Egg Activated! φ-resonance detected!');
}

/**
 * Initialize All Functions
 */
function init() {
    // Core functionality
    initNavbarScroll();
    initMobileMenu();
    initSmoothScroll();
    initScrollAnimations();
    updateActiveNavLink();

    // Interactive features
    initInnovationCards();
    animateStats();
    initCodeCopy();

    // Optional features
    logPerformance();
    initKonamiCode();

    // Log initialization
    console.log('%c🚀 MYSTIC v3.0 - Loaded Successfully', 'color: #0066ff; font-size: 16px; font-weight: bold;');
    console.log('%cZero drift. Unlimited horizon. 100% accuracy.', 'color: #6366f1; font-size: 12px;');
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Export for potential module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        init,
        initNavbarScroll,
        initMobileMenu,
        initSmoothScroll,
        initScrollAnimations,
        initInnovationCards,
        animateStats
    };
}
