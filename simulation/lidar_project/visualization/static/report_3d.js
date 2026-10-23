/* Offline, display-only adapters for saved report geometry and measured maps.
 * No scans, ray queries, planning, or map reconstruction occur in this file.
 * Plotly and the JSON payload are embedded by the report presentation.
 */
(function () {
  'use strict';

  const byId = id => document.getElementById(id);
  const C = {
    ink: '#223746', grid: '#E5EBEF', blue: '#126FA6', red: '#C63E48',
    target: '#E2BB78', panel: '#8997A4', path: '#C8753F', route: '#8799A5',
    sector: '#8DBCD2', background: '#FFFFFF'
  };
  const fovCamera = {eye: {x: -1.05, y: -1.2, z: .9}, up: {x: 0, y: 0, z: 1}};
  const mapCamera = {eye: {x: -1.1, y: -1.55, z: 1.15}, up: {x: 0, y: 0, z: 1}};
  const config = {responsive: true, displaylogo: false, scrollZoom: true};
  let source, mapStep = 0, playing = false, timer = null;
  let mapQueue = Promise.resolve(), mapRevision = 0, fovRevision = 0;
  let resizeObserver, intersectionObserver;

  const finite = v => typeof v === 'number' && Number.isFinite(v);
  const fmt = (v, digits = 1) => finite(v) ? v.toFixed(digits) : '—';
  const pct = v => finite(v) ? (100 * v).toFixed(1) + '%' : '—';
  const coordinates = points => ({
    x: points.map(p => p[0]), y: points.map(p => p[1]), z: points.map(p => p[2])
  });
  function text(id, value) { if (byId(id)) byId(id).textContent = value; }
  function bind(id, event, fn) { if (byId(id)) byId(id).addEventListener(event, fn); }
  function line(points, color, width, name, dash = 'solid', legend = false) {
    return {type: 'scatter3d', mode: 'lines', ...coordinates(points), name,
      line: {color, width, dash}, showlegend: legend, hoverinfo: 'skip'};
  }
  function separatedLines(segments, color, width, name, dash = 'solid') {
    const points = [];
    segments.forEach(segment => points.push(...segment, [null, null, null]));
    return line(points, color, width, name, dash, true);
  }
  function mesh(vertices, faces, color, name, opacity) {
    const triangles = [];
    faces.forEach(face => {
      for (let k = 1; k + 1 < face.length; k++) triangles.push([face[0], face[k], face[k + 1]]);
    });
    return {type: 'mesh3d', ...coordinates(vertices), i: triangles.map(f => f[0]),
      j: triangles.map(f => f[1]), k: triangles.map(f => f[2]), color, opacity,
      flatshading: true, name, hovertemplate: name + '<extra>explanatory scene geometry</extra>',
      showlegend: true, lighting: {ambient: .75, diffuse: .55, specular: .05}};
  }
  function axis(title, range) {
    return {title: {text: title, font: {size: 11}}, range, gridcolor: C.grid,
      zerolinecolor: C.grid, showbackground: true, backgroundcolor: '#FAFCFD',
      tickfont: {size: 10}, showspikes: false};
  }
  function layout(element, scene, legend = true) {
    return {autosize: true, height: Math.max(360, element.clientHeight || 460),
      paper_bgcolor: C.background, font: {family: 'Arial, sans-serif', color: C.ink, size: 11},
      scene, margin: {l: 0, r: 10, t: 10, b: legend ? 54 : 10},
      legend: {orientation: 'h', x: 0, y: -.04, font: {size: 11}, itemsizing: 'constant'}};
  }

  function selectedPanel() {
    const mode = byId('fov-mode') ? byId('fov-mode').value : 'panel_present';
    const id = ({panel: 'panel_present', no_panel: 'panel_absent'})[mode] || mode;
    return source.fov.panels.find(p => p.id === id) || source.fov.panels[source.fov.panels.length - 1];
  }
  function worldDirection(sensor, azimuth, elevation, distance) {
    const a = azimuth * Math.PI / 180, e = elevation * Math.PI / 180;
    const local = [Math.cos(e) * Math.cos(a), Math.cos(e) * Math.sin(a), Math.sin(e)];
    return sensor.position.map((value, row) => value + distance *
      sensor.rotation_world_from_sensor[row].reduce((sum, r, col) => sum + r * local[col], 0));
  }
  function sectorGeometry(fov) {
    const o = fov.sensor.position, h = fov.hfov_deg, v = fov.vfov_deg, r = fov.display_range_m;
    const arc = (varying, fixed, horizontal) => Array.from({length: 61}, (_, j) => {
      const angle = varying[0] + (varying[1] - varying[0]) * j / 60;
      return worldDirection(fov.sensor, horizontal ? angle : fixed, horizontal ? fixed : angle, r);
    });
    const lines = [arc(h, v[0], true), arc(h, v[1], true), arc(v, h[0], false), arc(v, h[1], false)];
    h.forEach(a => v.forEach(e => lines.push([o, worldDirection(fov.sensor, a, e, r)])));
    const vertices = [o], faces = [];
    // Two thin angular boundary sheets describe coverage, not measured surfaces.
    v.forEach(e => {
      const start = vertices.length;
      arc(h, e, true).forEach(p => vertices.push(p));
      for (let j = 0; j < 60; j++) faces.push([0, start + j, start + j + 1]);
    });
    return {lines, vertices, faces};
  }
  async function drawFov() {
    if (!byId('fov-plot') || !source.fov) return;
    const fov = source.fov, panel = selectedPanel(), origin = fov.sensor.position;
    const sector = sectorGeometry(fov), traces = [
      {...mesh(sector.vertices, sector.faces, C.sector, 'Nominal angular FOV', .1), hoverinfo: 'skip', hovertemplate: undefined},
      {...separatedLines(sector.lines, '#7EADC5', 1.5, 'FOV boundary'), showlegend: false}
    ];
    let obstacleLegend = false;
    panel.objects.forEach(object => {
      const target = object.role === 'target';
      const item = mesh(object.vertices, object.faces, target ? C.target : C.panel,
        target ? 'Target surface' : 'Fixed obstacle geometry', target ? .38 : .25);
      item.showlegend = target || !obstacleLegend;
      item.hovertemplate = (object.label || 'Fixed scene surface') + '<extra>explanatory scene geometry</extra>';
      if (!target) obstacleLegend = true;
      traces.push(item);
    });
    const visible = [], blocked = [], continuations = [], hits = [];
    panel.rays.forEach(ray => {
      if (ray.visible) visible.push([origin, ray.target_point]);
      else {
        blocked.push([origin, ray.nearest_point]);
        continuations.push([ray.nearest_point, ray.target_point]);
        hits.push(ray.nearest_point);
      }
    });
    if (visible.length) traces.push(separatedLines(visible, C.blue, 4.5, 'Visible target rays'));
    if (blocked.length) traces.push(separatedLines(blocked, C.red, 4.5, 'Nearest opaque hits'));
    if (continuations.length) traces.push(separatedLines(continuations, C.red, 4.5, 'Occluded continuation', 'dash'));
    if (hits.length) traces.push({type: 'scatter3d', mode: 'markers', ...coordinates(hits),
      marker: {size: 4, color: C.red, symbol: 'x'}, name: 'Blocking hit', showlegend: false, hoverinfo: 'skip'});
    traces.push({type: 'scatter3d', mode: 'markers+text', ...coordinates([origin]),
      marker: {size: 5.5, color: C.ink, symbol: 'diamond'}, text: ['LiDAR'],
      textposition: 'top center', name: 'Sensor', showlegend: false,
      hovertemplate: 'Saved sensor pose<br>x=%{x:.2f} m · y=%{y:.2f} m · z=%{z:.2f} m<extra></extra>'});
    ['xₛ', 'yₛ', 'zₛ'].forEach((letter, col) => {
      const endpoint = origin.map((value, row) => value + .55 * fov.sensor.rotation_world_from_sensor[row][col]);
      traces.push({...line([origin, endpoint], C.ink, 2, letter), mode: 'lines+text',
        text: ['', letter], textposition: 'top center', textfont: {size: 11}});
    });
    const bounds = sector.lines.flat().concat(panel.objects.flatMap(o => o.vertices));
    const ranges = [0, 1, 2].map(col => [Math.min(...bounds.map(p => p[col])) - .2,
      Math.max(...bounds.map(p => p[col])) + .2]);
    ranges[2][0] = Math.min(-.1, ranges[2][0]);
    const scene = {aspectmode: 'data', dragmode: 'orbit', camera: fovCamera,
      uirevision: 'fov-' + fovRevision, xaxis: axis('World x (m)', ranges[0]),
      yaxis: axis('World y (m)', ranges[1]), zaxis: axis('World z (m)', ranges[2])};
    const audit = panel.audit;
    text('fov-counts', `${audit.visible_target_echoes.toLocaleString()} visible · ` +
      `${audit.blocked_target_echoes.toLocaleString()} blocked · ` +
      `${audit.target_opportunities.toLocaleString()} target intersections in the complete ray audit`);
    text('fov-note', `Same saved sensor pose and angular grid. ${panel.rays.length} representative rays are drawn. ` +
      `The FOV is clipped to ${fmt(fov.display_range_m, 0)} m for display; the sensor range gate is ` +
      `${fmt(fov.min_range_m, 1)}–${fmt(fov.max_range_m, 0)} m. Existing surfaces are drawn translucent to expose the rays and are modeled as opaque. The weather comparison changes only atmospheric attenuation.`);
    await Plotly.react(byId('fov-plot'), traces, layout(byId('fov-plot'), scene), config);
    byId('fov-plot').dataset.ready = 'true';
    byId('fov-plot').dataset.panel = panel.id;
  }

  function selectedMap() {
    const id = byId('map-case') ? byId('map-case').value : 'baseline';
    return {id: source.mapping[id] ? id : 'baseline', value: source.mapping[id] || source.mapping.baseline};
  }
  function floorIndex(values, t) {
    let lo = 0, hi = values.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (values[mid] <= t + 1e-8) lo = mid + 1; else hi = mid; }
    return lo - 1;
  }
  function robotBody(position, radius, height) {
    const vertices = [], faces = [], n = 20;
    for (const z of [0, height]) for (let j = 0; j < n; j++) {
      const a = 2 * Math.PI * j / n;
      vertices.push([position[0] + radius * Math.cos(a), position[1] + radius * Math.sin(a), z]);
    }
    vertices.push([position[0], position[1], 0], [position[0], position[1], height]);
    for (let j = 0; j < n; j++) {
      const k = (j + 1) % n;
      faces.push([j, k, n + k], [j, n + k, n + j], [2 * n, k, j], [2 * n + 1, n + j, n + k]);
    }
    return {...mesh(vertices, faces, C.path, 'Robot body', .58), showlegend: false,
      hovertemplate: 'Saved robot position<extra>cylindrical collision envelope</extra>'};
  }
  function totalTimeline() {
    return [...new Set(Object.values(source.mapping).flatMap(c => c.times || []))].sort((a, b) => a - b);
  }
  async function drawMap() {
    if (!byId('map-plot') || !source.mapping) return;
    const selected = selectedMap(), c = selected.value, timeline = totalTimeline();
    mapStep = Math.max(0, Math.min(mapStep, timeline.length - 1));
    const t = timeline[mapStep], k = Math.max(0, floorIndex(c.times, t));
    const scanStep = floorIndex(c.scan_times, t), ended = t >= c.times[c.times.length - 1] - 1e-8;
    const showGround = !byId('map-ground') || byId('map-ground').checked;
    const visible = c.points.filter(p => p[3] <= scanStep && (showGround || p[2] > .12));
    const curve = c.full_map_curve.filter(r => r.step <= scanStep).at(-1);
    const trajectory = c.trajectory.slice(0, k + 1), position = trajectory.at(-1);
    const traces = [{type: 'scatter3d', mode: 'markers', ...coordinates(visible),
      customdata: visible.map(p => p[3]), name: 'Measured surface samples', showlegend: false,
      marker: {size: .75, opacity: .72, color: visible.map(p => p[2]), cmin: 0, cmax: 3,
        colorscale: [[0, '#6D9EB5'], [.2, '#3D9C94'], [.5, '#E1BB70'], [1, '#C46C5D']],
        colorbar: {title: {text: 'Height (m)', font: {size: 10}}, thickness: 9, len: .65,
          tickfont: {size: 10}, outlinewidth: 0}},
      hovertemplate: 'Measured sample<br>x=%{x:.2f} m · y=%{y:.2f} m<br>z=%{z:.2f} m · first seen at step %{customdata}<extra></extra>'}];
    traces.push({...line(c.reference, C.route, 2, 'Shared planned route', 'dash', true),
      hovertemplate: 'Known-map reference<extra></extra>', hoverinfo: undefined});
    traces.push(line(trajectory.map(p => [p[0], p[1], .045]), C.path, 4, 'Executed path so far', 'solid', true));
    if (position) {
      traces.push(robotBody(position, c.robot_radius_m || .35, c.robot_height_m || 1));
      const angle = c.theta_rad[k] || 0;
      traces.push(line([[position[0], position[1], .37],
        [position[0] + .75 * Math.cos(angle), position[1] + .75 * Math.sin(angle), .37]], C.path, 5, 'Robot heading'));
      const height = finite(c.mount_height_m) ? c.mount_height_m : .6;
      traces.push({type: 'scatter3d', mode: 'markers', x: [position[0]], y: [position[1]], z: [height],
        marker: {size: 4, color: C.ink, symbol: 'diamond'}, name: 'LiDAR location', showlegend: true,
        hovertemplate: 'Sensor mount<br>x=%{x:.2f} m · y=%{y:.2f} m · z=%{z:.2f} m<extra></extra>'});
    }
    const start = c.reference[0], goal = c.reference.at(-1);
    traces.push({type: 'scatter3d', mode: 'markers+text', ...coordinates([start, goal]),
      marker: {size: [3, 5], color: [C.route, C.ink], symbol: ['circle', 'cross']},
      text: ['Start', 'Goal'], textposition: 'top center', showlegend: false, hoverinfo: 'text'});
    const extent = c.extent_m || [-2, 21, -6, 7];
    const scene = {aspectmode: 'data', dragmode: 'orbit', camera: mapCamera,
      uirevision: 'map-' + mapRevision, xaxis: axis('World x (m)', extent.slice(0, 2)),
      yaxis: axis('World y (m)', extent.slice(2, 4)), zaxis: axis('World z (m)', [-.08, 3])};
    text('map-time-label', `${fmt(c.times[k], 1)} s${ended ? ' · final robot pose' : ''}`);
    const count = curve ? curve.points : 0;
    const observed = curve && finite(curve.observed_fraction) ? ` · ${pct(curve.observed_fraction)} endpoint cells observed` : '';
    text('map-counts', `${Number(count).toLocaleString()} measured map voxels${observed}`);
    text('map-note', `${visible.length.toLocaleString()} saved samples shown from the map available by this time. ` +
      (scanStep >= 0 ? `${ended ? 'Last' : 'Latest'} scan: ${fmt(c.scan_times[scanStep], 1)} s. ` : 'No scan recorded yet. ') +
      (showGround ? '' : 'Low returns at or below 0.12 m are hidden. ') +
      'The view contains measured endpoints, not an inferred free-space map; robot position is known.');
    if (byId('map-time')) { byId('map-time').max = timeline.length - 1; byId('map-time').value = mapStep; }
    await Plotly.react(byId('map-plot'), traces, layout(byId('map-plot'), scene), config);
    byId('map-plot').dataset.ready = 'true';
    byId('map-plot').dataset.case = selected.id;
    byId('map-plot').dataset.time = t;
    byId('map-plot').dataset.scanStep = scanStep;
    byId('map-plot').dataset.displayedPoints = visible.length;
    byId('map-plot').dataset.futurePoints = visible.filter(p => p[3] > scanStep).length;
  }
  function fail(error) {
    pause();
    text('map-note', 'The saved 3D view could not be displayed: ' + error.message);
    document.documentElement.dataset.report3dError = error.message;
  }
  function enqueueMap() {
    mapQueue = mapQueue.catch(fail).then(drawMap);
    return mapQueue;
  }
  function pause() {
    playing = false;
    clearTimeout(timer);
    text('map-play', 'Play');
    if (byId('map-play')) byId('map-play').setAttribute('aria-pressed', 'false');
  }
  function schedule() {
    if (!playing) return;
    const timeline = totalTimeline();
    if (mapStep >= timeline.length - 1) return pause();
    const speed = byId('map-speed') ? (+byId('map-speed').value || 2) : 2;
    const delay = Math.max(40, 1000 * (timeline[mapStep + 1] - timeline[mapStep]) / speed);
    timer = setTimeout(() => { mapStep++; enqueueMap().then(schedule); }, delay);
  }
  function resize() {
    ['fov-plot', 'map-plot'].forEach(id => {
      const node = byId(id);
      if (node && node.data && node.clientWidth > 0) Plotly.Plots.resize(node);
    });
  }
  async function init(data) {
    source = data;
    // Land on the complete saved map; Play deliberately restarts at the first scan.
    if (source.mapping) mapStep = Math.max(0, totalTimeline().length - 1);
    if (typeof Plotly === 'undefined') throw new Error('The embedded Plotly library is missing.');
    bind('fov-mode', 'change', () => drawFov().catch(fail));
    bind('fov-reset', 'click', () => { fovRevision++; drawFov().catch(fail); });
    bind('map-case', 'change', () => { pause(); enqueueMap(); });
    bind('map-time', 'input', () => { pause(); mapStep = +byId('map-time').value; enqueueMap(); });
    bind('map-ground', 'change', enqueueMap);
    bind('map-reset', 'click', () => { mapRevision++; enqueueMap(); });
    bind('map-play', 'click', () => {
      if (playing) return pause();
      if (mapStep >= totalTimeline().length - 1) mapStep = 0;
      playing = true;
      text('map-play', 'Pause');
      byId('map-play').setAttribute('aria-pressed', 'true');
      enqueueMap().then(schedule);
    });
    bind('map-speed', 'change', () => { if (playing) { clearTimeout(timer); schedule(); } });
    if (byId('map-time')) {
      byId('map-time').min = 0;
      byId('map-time').step = 1;
      byId('map-time').setAttribute('aria-label', 'Saved navigation time');
    }
    window.addEventListener('resize', resize);
    if (typeof ResizeObserver !== 'undefined') {
      resizeObserver = new ResizeObserver(resize);
      ['fov-plot', 'map-plot'].forEach(id => { if (byId(id)) resizeObserver.observe(byId(id)); });
    }
    if (typeof IntersectionObserver !== 'undefined') {
      intersectionObserver = new IntersectionObserver(entries => { if (entries.some(e => e.isIntersecting)) resize(); });
      ['fov-plot', 'map-plot'].forEach(id => { if (byId(id)) intersectionObserver.observe(byId(id)); });
    }
    await Promise.all([drawFov(), enqueueMap()]);
    document.documentElement.dataset.report3dReady = 'true';
  }

  window.Report3D = {init, resize, pause};
}());
