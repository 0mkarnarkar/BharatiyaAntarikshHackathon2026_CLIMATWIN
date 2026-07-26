/* ═══════════════════════════════════════════════════════════════
   CLIMATWIN — Apple-Style GSAP + Three.js Engine
   ═══════════════════════════════════════════════════════════════ */

gsap.registerPlugin(ScrollTrigger, ScrollToPlugin);

// ── Smooth Scroll (Lenis) ──
const lenis = new Lenis({
  duration: 3.5, // Much slower smooth duration
  easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
  smooth: true,
  wheelMultiplier: 0.5, // Slows down the physical scroll wheel distance
});

lenis.on('scroll', ScrollTrigger.update);

gsap.ticker.add((time) => {
  lenis.raf(time * 1000);
});
gsap.ticker.lagSmoothing(0);

// ────────────────────────────────────────────────────────────────
// 1. THREE.JS — THERMAL DATA SPHERE
// ────────────────────────────────────────────────────────────────
(function initThreeJS() {
  const canvas = document.getElementById('hero-canvas');
  if (!canvas) return;

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(0, 0, 80);

  // Particle Setup
  const COUNT = 5000;
  const geo = new THREE.BufferGeometry();
  const positions = new Float32Array(COUNT * 3);
  const colors = new Float32Array(COUNT * 3);
  const sizes = new Float32Array(COUNT);

  const palette = [
    [0.05, 0.45, 0.95], [0.10, 0.60, 0.98],
    [0.20, 0.82, 0.90], [0.85, 0.85, 0.95]
  ];

  for (let i = 0; i < COUNT; i++) {
    const i3 = i * 3;
    // Even sphere distribution
    const u = Math.random();
    const v = Math.random();
    const theta = u * Math.PI * 2;
    const phi = Math.acos(2 * v - 1);
    const r = 32 + Math.random() * 5;
    positions[i3]     = r * Math.sin(phi) * Math.cos(theta);
    positions[i3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    positions[i3 + 2] = r * Math.cos(phi);

    const c = palette[Math.floor(Math.random() * palette.length)];
    colors[i3] = c[0]; colors[i3 + 1] = c[1]; colors[i3 + 2] = c[2];
    sizes[i] = Math.random() * 2.5 + 1.0;
  }

  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geo.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

  const mat = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 } },
    vertexShader: [
      'attribute float size;',
      'attribute vec3 color;',
      'varying vec3 vColor;',
      'varying float vAlpha;',
      'uniform float uTime;',
      'void main() {',
      '  vColor = color;',
      '  vec3 p = position;',
      '  p += normalize(p) * sin(uTime * 0.4 + length(p.xy) * 0.08) * 3.0;',
      '  vec4 mv = modelViewMatrix * vec4(p, 1.0);',
      '  gl_PointSize = size * (140.0 / -mv.z);',
      '  gl_Position = projectionMatrix * mv;',
      '  vAlpha = smoothstep(-200.0, -10.0, mv.z) * 0.55;',
      '}'
    ].join('\n'),
    fragmentShader: [
      'varying vec3 vColor;',
      'varying float vAlpha;',
      'void main() {',
      '  float d = length(gl_PointCoord - vec2(0.5));',
      '  if (d > 0.5) discard;',
      '  float a = pow(1.0 - d * 2.0, 1.5) * vAlpha;',
      '  gl_FragColor = vec4(vColor, a);',
      '}'
    ].join('\n'),
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  });

  const points = new THREE.Points(geo, mat);
  scene.add(points);

  // Mouse tracking for subtle parallax and cursor glow
  let mouseX = 0, mouseY = 0;
  let idleTimer;
  const cursorGlow = document.getElementById('cursorGlow');
  
  window.addEventListener('mousemove', function(e) {
    mouseX = (e.clientX / window.innerWidth - 0.5) * 2;
    mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
    
    if (cursorGlow) {
      cursorGlow.style.left = e.clientX + 'px';
      cursorGlow.style.top = e.clientY + 'px';
    }
    
    if (!document.body.classList.contains('recording-mode')) {
      document.body.style.cursor = 'default';
      if(cursorGlow) cursorGlow.style.opacity = '1';
      
      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => {
        document.body.style.cursor = 'none';
        if(cursorGlow) cursorGlow.style.opacity = '0';
      }, 2000); // Hide cursor after 2 seconds idle
    }
  });

  window.addEventListener('resize', function() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  let t = 0;
  function animate() {
    requestAnimationFrame(animate);
    t += 0.01;
    mat.uniforms.uTime.value = t;
    // Slow base rotation + mouse parallax
    points.rotation.y = t * 0.08 + mouseX * 0.15;
    points.rotation.x = t * 0.04 + mouseY * 0.1;
    renderer.render(scene, camera);
  }
  animate();
})();

// ────────────────────────────────────────────────────────────────
// 2. GSAP — SCROLL ANIMATIONS
// ────────────────────────────────────────────────────────────────

// ── Nav Visibility ──
ScrollTrigger.create({
  start: 'top -80',
  onUpdate: function(self) {
    var nav = document.getElementById('nav');
    if (self.direction === -1 && self.progress > 0) {
      nav.classList.add('visible');
    } else if (self.progress === 0 || self.direction === 1) {
      nav.classList.remove('visible');
    }
  }
});

// ── Progress Bar ──
gsap.to('#progressFill', {
  width: '100%',
  ease: 'none',
  scrollTrigger: { scrub: 0.3 }
});

// ── Hero Parallax ──
gsap.to('.hero-content', {
  y: -100,
  opacity: 0,
  ease: 'none',
  scrollTrigger: {
    trigger: '.hero-section',
    start: 'top top',
    end: 'bottom top',
    scrub: true
  }
});

// ── Stat Counter Animation ──
ScrollTrigger.create({
  trigger: '.hero-stats',
  start: 'top 85%',
  once: true,
  onEnter: function() {
    document.querySelectorAll('.stat-val').forEach(function(el) {
      var target = parseFloat(el.getAttribute('data-target'));
      if (isNaN(target)) return;
      var isFloat = target < 10 && target % 1 !== 0;
      var obj = { val: 0 };
      gsap.to(obj, {
        val: target,
        duration: 2.5,
        ease: 'power2.out',
        onUpdate: function() {
          el.textContent = isFloat ? obj.val.toFixed(1) : Math.round(obj.val);
        }
      });
    });
  }
});

// ── General Fade-Up Animations ──
var fadeTargets = '.problem-card, .pipeline-step, .science-card, .stack-row, .impact-card, .cta-block';
gsap.utils.toArray(fadeTargets).forEach(function(el, i) {
  gsap.from(el, {
    y: 40,
    opacity: 0,
    duration: 0.8,
    delay: i % 4 * 0.1,
    ease: 'power3.out',
    scrollTrigger: {
      trigger: el,
      start: 'top 88%',
      toggleActions: 'play none none none'
    }
  });
});

// ── Section Headers ──
gsap.utils.toArray('.section-header').forEach(function(el) {
  gsap.from(el, {
    y: 30,
    opacity: 0,
    duration: 1,
    ease: 'power3.out',
    scrollTrigger: {
      trigger: el,
      start: 'top 85%',
      toggleActions: 'play none none none'
    }
  });
});

// ── Horizontal Scroll Features ──
(function initHorizontalScroll() {
  var track = document.querySelector('.features-track');
  var pin = document.querySelector('.features-pin');
  if (!track || !pin) return;

  gsap.to(track, {
    x: function() { return -(track.scrollWidth - window.innerWidth); },
    ease: 'none',
    scrollTrigger: {
      id: 'features-scroll',
      trigger: pin,
      pin: true,
      scrub: 1,
      start: 'top top',
      end: function() { return '+=' + (track.scrollWidth * 1.8); }, // Increased scroll distance to slow down horizontal movement
      invalidateOnRefresh: true
    }
  });
})();

// ── Impact Counter ──
ScrollTrigger.create({
  trigger: '.impact-grid',
  start: 'top 80%',
  once: true,
  onEnter: function() {
    document.querySelectorAll('.impact-val').forEach(function(el) {
      var target = parseFloat(el.getAttribute('data-target'));
      if (isNaN(target)) return;
      // Handle the "₹0" special case
      if (target === 0) { el.textContent = '₹0'; return; }
      var isFloat = target < 1;
      var obj = { val: 0 };
      gsap.to(obj, {
        val: target,
        duration: 2,
        ease: 'power2.out',
        onUpdate: function() {
          el.textContent = isFloat ? obj.val.toFixed(2) : Math.round(obj.val);
        }
      });
    });
  }
});

// ── Smooth Scroll for Nav Links ──
document.querySelectorAll('a[href^="#"]').forEach(function(a) {
  a.addEventListener('click', function(e) {
    e.preventDefault();
    var target = a.getAttribute('href');
    lenis.scrollTo(target, { duration: 1.5, easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)) });
  });
});

// ── Cinematic Auto-Scroll for Recording (Press 'R') ──
window.addEventListener('keydown', (e) => {
  if (e.key.toLowerCase() === 'r' && !document.body.classList.contains('recording-mode')) {
    const overlay = document.getElementById('countdownOverlay');
    const countText = document.getElementById('countdownText');
    if(!overlay) return;

    console.log('🎥 Initiating Cinematic Auto-Scroll...');
    document.body.classList.add('recording-mode');
    const cg = document.getElementById('cursorGlow');
    if (cg) cg.style.opacity = '0';
    
    // Jump to top immediately
    lenis.scrollTo(0, { immediate: true });
    
    overlay.classList.add('active');
    let count = 3;
    countText.textContent = count;
    
    const interval = setInterval(() => {
      count--;
      if (count > 0) {
        countText.textContent = count;
      } else {
        clearInterval(interval);
        overlay.classList.remove('active');
        
        setTimeout(async () => {
          const getWaypoints = () => {
            const points = [];
            points.push(document.getElementById('problem').offsetTop);
            points.push(document.getElementById('pipeline').offsetTop);
            
            const features = document.getElementById('features');
            const track = document.querySelector('.features-track');
            const fTop = features.offsetTop;
            const fScrollDist = track.scrollWidth * 1.8;
            
            // Pauses through the horizontal section
            points.push(fTop);
            points.push(fTop + fScrollDist * 0.25);
            points.push(fTop + fScrollDist * 0.50);
            points.push(fTop + fScrollDist * 0.75);
            points.push(fTop + fScrollDist);
            
            points.push(document.getElementById('science').offsetTop);
            points.push(document.getElementById('stack').offsetTop);
            points.push(document.getElementById('impact').offsetTop);
            points.push(document.documentElement.scrollHeight - window.innerHeight);
            return points;
          };
          
          const waypoints = getWaypoints();
          const pauseDuration = 4000; // 4 seconds to read
          const scrollDuration = 1.5; // 1.5 seconds to slide to next
          
          for (let i = 0; i < waypoints.length; i++) {
            if (!document.body.classList.contains('recording-mode')) break;
            
            // Pause
            await new Promise(r => setTimeout(r, pauseDuration));
            if (!document.body.classList.contains('recording-mode')) break;
            
            // Slide
            lenis.scrollTo(waypoints[i], {
              duration: scrollDuration,
              easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
              lock: true
            });
            
            // Wait for slide to finish before next loop
            await new Promise(r => setTimeout(r, scrollDuration * 1000));
          }
          
          if (document.body.classList.contains('recording-mode')) {
             document.body.classList.remove('recording-mode');
             lenis.scrollTo(window.scrollY, { lock: false });
          }
        }, 500);
      }
    }, 1000);
  }
  
  // Press Escape to cancel recording mode
  if (e.key === 'Escape' && document.body.classList.contains('recording-mode')) {
    document.body.classList.remove('recording-mode');
    lenis.scrollTo(window.scrollY, { lock: false }); // unlock
    const overlay = document.getElementById('countdownOverlay');
    if(overlay) {
      overlay.classList.remove('active');
      // Hack to clear interval if we escaped during countdown
      let id = window.setTimeout(function() {}, 0);
      while (id--) { window.clearTimeout(id); }
    }
  }
});

// ── Vanilla Tilt ──
if (typeof VanillaTilt !== 'undefined') {
  VanillaTilt.init(document.querySelectorAll(".problem-card, .science-card, .impact-card, .cta-block, .mock-window"), {
    max: 5,
    speed: 400,
    glare: true,
    "max-glare": 0.15,
    scale: 1.02
  });
}

// ── Feature Mock UI Animations ──
gsap.utils.toArray('.mock-bars').forEach(function(barsContainer) {
  var fills = barsContainer.querySelectorAll('.bar-fill');
  gsap.from(fills, {
    width: 0,
    duration: 1.5,
    ease: 'power3.out',
    stagger: 0.15,
    scrollTrigger: {
      trigger: barsContainer,
      start: 'left center', // for horizontal scroll
      containerAnimation: gsap.getById('features-scroll'),
      toggleActions: 'play none none none'
    }
  });
});

gsap.utils.toArray('.mock-sliders').forEach(function(sliderContainer) {
  var fills = sliderContainer.querySelectorAll('.slider-fill');
  var thumbs = sliderContainer.querySelectorAll('.slider-thumb');
  
  gsap.from(fills, {
    width: 0,
    duration: 1.5,
    ease: 'power3.out',
    stagger: 0.2,
    scrollTrigger: {
      trigger: sliderContainer,
      start: 'left center',
      containerAnimation: gsap.getById('features-scroll'),
      toggleActions: 'play none none none'
    }
  });
  
  gsap.from(thumbs, {
    left: 0,
    duration: 1.5,
    ease: 'power3.out',
    stagger: 0.2,
    scrollTrigger: {
      trigger: sliderContainer,
      start: 'left center',
      containerAnimation: gsap.getById('features-scroll'),
      toggleActions: 'play none none none'
    }
  });
});

console.log('🌡️ CLIMATWIN — Premium Engine Loaded');
