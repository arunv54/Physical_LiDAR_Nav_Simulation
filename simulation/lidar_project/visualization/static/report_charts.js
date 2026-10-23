/* Offline report charts: saved observations and explicitly labelled theory. */
(function () {
  'use strict';
  const C = {blue: '#2579A8', orange: '#C8753F', teal: '#3D9C94', ink: '#243844', muted: '#61727B', grid: '#E6ECEF'};
  const CONFIG = {responsive: true, displaylogo: false, scrollZoom: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d']};
  let DATA;
  const rendered = new Set();
  const byId = id => document.getElementById(id);
  const finite = value => value !== null && value !== undefined && Number.isFinite(Number(value));
  const format = (value, digits = 2) => finite(value) ? Number(value).toLocaleString('en-US', {minimumFractionDigits: digits, maximumFractionDigits: digits}) : '—';
  const integer = value => format(value, 0);
  const percent = value => finite(value) ? format(Number(value) * 100, 1) + '%' : '—';
  const text = (id, value) => {if (byId(id)) byId(id).textContent = value;};
  const lines = (id, values) => {
    const host = byId(id);
    if (!host) return;
    host.replaceChildren(...values.map(value => {
      const line = document.createElement('span');
      line.style.display = 'block';
      line.textContent = value;
      return line;
    }));
  };
  const axis = (title, extra = {}) => ({title: {text: title, standoff: 12},
    automargin: true, gridcolor: C.grid, zerolinecolor: '#ACBBC2', linecolor: '#C6D1D6',
    tickfont: {size: 11}, ...extra});
  const layout = (xlabel, ylabel, extra = {}) => ({
    autosize: true, height: 320, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: '#FFFFFF',
    font: {family: 'Arial, sans-serif', size: 12, color: C.ink},
    margin: {l: 65, r: 24, t: 30, b: 66},
    xaxis: axis(xlabel), yaxis: axis(ylabel),
    legend: {orientation: 'h', x: 0, y: 1.16, font: {size: 11}},
    hoverlabel: {font: {size: 12}, bgcolor: '#FFFFFF'},
    ...extra
  });
  function plot(id, traces, spec) {
    const host = byId(id);
    if (!host) return Promise.resolve();
    rendered.add(id);
    return Plotly.react(host, traces, spec, CONFIG);
  }
  function optionList(id, values, selected) {
    const host = byId(id);
    if (!host) return;
    host.replaceChildren(...values.map(([value, label]) => {
      const option = document.createElement('option');
      option.value = String(value); option.textContent = label;
      return option;
    }));
    host.value = String(selected);
  }
  const errorBars = (centers, lows, highs) => ({type: 'data', symmetric: false,
    array: highs.map((high, i) => Math.max(0, high - centers[i])),
    arrayminus: lows.map((low, i) => Math.max(0, centers[i] - low)), thickness: 1.25, width: 4});
  const normalCDF = value => {
    const absolute = Math.abs(value), t = 1 / (1 + 0.2316419 * absolute);
    const tail = Math.exp(-absolute * absolute / 2) / Math.sqrt(2 * Math.PI) *
      t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    return value < 0 ? tail : 1 - tail;
  };

  async function renderPower() {
    const get = (id, fallback) => byId(id) ? Number(byId(id).value) : fallback;
    const range = get('power-range', 5), rho = get('power-rho', .3);
    const kappa = get('power-kappa', .0002), angle = get('power-angle', 0);
    text('power-range-value', format(range, 1) + ' m');
    text('power-rho-value', format(rho, 2));
    text('power-kappa-value', format(kappa, 4) + ' m⁻¹');
    text('power-angle-value', format(angle, 0) + '°');
    const ratio = (r, reflectance, extinction, degrees) =>
      reflectance / .3 * Math.cos(degrees * Math.PI / 180) * (5 / r) ** 2 *
      Math.exp(-2 * (extinction * r - .0002 * 5));
    const xs = Array.from({length: 160}, (_, i) => 1 + 19 * i / 159);
    const value = ratio(range, rho, kappa, angle);
    const number = value >= .001 && value < 1000 ? format(value, 3) : value.toExponential(3);
    lines('power-readout', [
      'Model prediction: relative returned power = ' + number + ' × the reference.',
      'Reference: 5 m, diffuse weight 0.30, normal incidence, κ = 0.0002 m⁻¹. Fixed optics; eligible, visible, beam-filling diffuse echo.',
      'These sliders evaluate Equation (7). They do not create measurements or rerun a navigation experiment.'
    ]);
    return plot('power-model-plot', [
      {x: xs, y: xs.map(r => ratio(r, .3, .0002, 0)), name: 'Reference surface / air',
        mode: 'lines', line: {color: '#9AA9B1', dash: 'dash', width: 2},
        hovertemplate: 'Range %{x:.2f} m<br>Reference %{y:.4g} ×<extra></extra>'},
      {x: xs, y: xs.map(r => ratio(r, rho, kappa, angle)), name: 'Selected model parameters',
        mode: 'lines', line: {color: C.blue, width: 3},
        hovertemplate: 'Range %{x:.2f} m<br>Predicted power %{y:.4g} ×<extra></extra>'},
      {x: [range], y: [value], name: 'Selected range', mode: 'markers',
        marker: {color: C.orange, size: 10, line: {color: '#FFFFFF', width: 2}},
        hovertemplate: 'Range %{x:.2f} m<br>Predicted power %{y:.4g} ×<extra></extra>'}
    ], layout('Range (m)', 'Relative returned power (log scale)', {
      yaxis: axis('Relative returned power', {type: 'log'}),
      shapes: [{type: 'line', x0: range, x1: range, y0: 0, y1: 1, yref: 'paper',
        line: {color: C.orange, dash: 'dot', width: 1}}]
    }));
  }

  async function renderThreshold() {
    const study = DATA.threshold, selection = byId('threshold-pose')?.value || 'all';
    const poses = study.views.filter(view => selection === 'all' || String(view.pose_id) === selection);
    const levels = [...new Set(study.settings.map(s => Math.round(s.mean_snr * 100) / 100))].sort((a, b) => a - b);
    const index = Math.max(0, Math.min(levels.length - 1, Number(byId('threshold-snr')?.value ?? 5)));
    const snr = levels[index];
    text('threshold-snr-value', 'Mean SNR ' + format(snr, 1));
    const xs = Array.from({length: 161}, (_, i) => 6 + 8 * i / 160);
    const traces = [{x: xs, y: xs.map(x => normalCDF(x - 10) * 100), mode: 'lines',
      name: 'Model Φ(SNR − 10)', line: {color: C.ink, width: 2},
      hovertemplate: 'Mean SNR %{x:.2f}<br>Predicted detection %{y:.2f}%<extra></extra>'}];
    const residuals = [], readout = [];
    for (const view of poses) {
      const rows = study.settings.filter(row => row.pose_id === view.pose_id).sort((a, b) => a.mean_snr - b.mean_snr);
      const color = [C.blue, C.orange, C.teal][view.pose_id % 3];
      const observed = rows.map(row => row.observed_pd * 100);
      const low = rows.map(row => row.wilson_low * 100), high = rows.map(row => row.wilson_high * 100);
      const offset = (view.pose_id - 1) * .045;
      const custom = rows.map(row => [row.mean_snr, row.detections, row.trials, row.expected_pd * 100,
        row.wilson_low * 100, row.wilson_high * 100, row.extinction_per_m]);
      traces.push({x: rows.map(row => row.mean_snr + offset), y: observed, mode: 'markers', name: view.label,
        marker: {color, size: rows.map(row => Math.abs(row.mean_snr - snr) < .01 ? 11 : 7),
          line: {color: '#FFFFFF', width: 1}}, error_y: {...errorBars(observed, low, high), color}, customdata: custom,
        hovertemplate: 'Mean SNR %{customdata[0]:.1f}<br>%{customdata[1]}/%{customdata[2]} detected<br>Observed %{y:.2f}%<br>Model %{customdata[3]:.2f}%<br>95% Wilson [%{customdata[4]:.2f}, %{customdata[5]:.2f}]%<br>κ=%{customdata[6]:.4f} m⁻¹<extra>%{fullData.name}</extra>'});
      const delta = rows.map(row => (row.observed_pd - row.expected_pd) * 100);
      residuals.push({x: rows.map(row => row.mean_snr + offset), y: delta, name: view.label, mode: 'markers',
        marker: {color, size: 7}, customdata: custom,
        error_y: {...errorBars(delta, rows.map(row => (row.wilson_low - row.expected_pd) * 100),
          rows.map(row => (row.wilson_high - row.expected_pd) * 100)), color},
        hovertemplate: 'Mean SNR %{customdata[0]:.1f}<br>Observed − predicted %{y:.3f} percentage points<extra>%{fullData.name}</extra>'});
      const chosen = rows.find(row => Math.abs(row.mean_snr - snr) < .01);
      if (chosen) readout.push(view.label + ': ' + integer(chosen.detections) + '/' + integer(chosen.trials) +
        ' detections (' + percent(chosen.observed_pd) + '); model ' + percent(chosen.expected_pd) +
        '; pointwise 95% Wilson [' + percent(chosen.wilson_low) + ', ' + percent(chosen.wilson_high) + '].');
    }
    readout.push('Every marker counts 1,000 independent actual scan trials, including misses. Horizontal offsets separate viewpoints; they do not change SNR. The shaded model transition is not a confidence interval.');
    lines('threshold-readout', readout);
    const shapes = [{type: 'rect', x0: 8.355146373, x1: 11.644853627, y0: 0, y1: 1, yref: 'paper',
      fillcolor: 'rgba(37,121,168,.07)', line: {width: 0}, layer: 'below'},
      {type: 'line', x0: snr, x1: snr, y0: 0, y1: 1, yref: 'paper', line: {color: C.orange, dash: 'dot', width: 1}}];
    return Promise.all([
      plot('threshold-plot', traces, layout('Mean echo power / noise RMS (SNR)', 'Detected / trials (%)', {
        xaxis: axis('Mean echo power / noise RMS (SNR)', {range: [5.8, 14.2]}),
        yaxis: axis('Detected / trials (%)', {range: [-3, 103]}), shapes})),
      plot('threshold-residual', residuals, layout('Mean echo power / noise RMS (SNR)', 'Observed − predicted (percentage points)', {
        height: 300, xaxis: axis('Mean echo power / noise RMS (SNR)', {range: [5.8, 14.2]}),
        shapes: [{type: 'line', x0: 5.8, x1: 14.2, y0: 0, y1: 0, line: {color: C.ink, width: 1, dash: 'dash'}}]
      }))
    ]);
  }

  const weatherIds = ['baseline', 'rain', 'fog'];
  const weatherColors = {baseline: C.blue, rain: C.teal, fog: C.orange};
  const weatherNames = {baseline: 'Clear air', rain: 'Rain 50 mm/h', fog: 'Fog κ = 0.6 m⁻¹'};
  const getCase = id => DATA.primary.cases.find(row => row.id === id);

  async function renderDependencies() {
    const cases = DATA.targets.cases;
    const selected = cases.find(row => row.id === (byId('dependency-case')?.value || 'reflectance_dark'));
    const pair = selected.id === 'baseline' ? [selected] : [cases.find(row => row.id === 'baseline'), selected];
    const labels = ['Target intersections', 'Visible target hits', 'In-range echoes', 'Detected returns'];
    const keys = ['target_opportunities', 'visible_target_echoes', 'eligible_echoes', 'detected_returns'];
    const stages = pair.map((item, index) => ({type: 'bar', name: item.label,
      x: labels, y: keys.map(key => item.replay[key]), marker: {color: index ? C.orange : C.blue},
      hovertemplate: '%{x}<br>%{y:,} target echoes<extra>%{fullData.name}</extra>'}));
    const traces = [];
    for (const [index, item] of pair.entries()) {
      traces.push({type: 'bar', x: [item.label], y: [1000 * item.replay.mean_eligible_power_w],
        marker: {color: index ? C.orange : C.blue}, name: item.label, showlegend: false,
        hovertemplate: '%{x}<br>%{y:.6f} mW eligible echo power<extra></extra>'});
      traces.push({type: 'bar', x: [item.label], y: [1000 * item.replay.range_rmse_m],
        marker: {color: index ? C.orange : C.blue}, name: item.label, showlegend: false, xaxis: 'x2', yaxis: 'y2',
        hovertemplate: '%{x}<br>%{y:.3f} mm conditional target RMS<extra></extra>'});
    }
    lines('dependency-summary', [selected.label + ': ' + selected.description,
      integer(selected.replay.eligible_echoes) + ' eligible target echoes → ' + integer(selected.replay.detected_returns) +
      ' detections (' + percent(selected.replay.observed_detection_probability) + ').',
      'Target settings are compared at the same 271 baseline snapshots. Material changes act on echo power; height, yaw and location can also change sampling, occlusion and range eligibility.']);
    text('dependency-note', 'Replays pool seeds 1–3 at identical saved baseline poses. Eligible power averages every eligible target echo before detector selection; RMS pools squared errors over detected target returns only. No target-change navigation claim is drawn from this comparison.');
    return Promise.all([
      plot('dependency-stage-plot', stages, layout('', 'Target echoes / returns', {barmode: 'group',
        margin: {l: 68, r: 16, t: 50, b: 78}, xaxis: axis('', {tickangle: -12})})),
      plot('dependency-power-plot', traces, layout('', '', {margin: {l: 62, r: 62, t: 47, b: 64},
        xaxis: axis('', {domain: [0, .39], tickangle: -10}),
        yaxis: axis('Eligible power (mW)', {rangemode: 'tozero'}),
        xaxis2: axis('', {domain: [.61, 1], anchor: 'y2', tickangle: -10}),
        yaxis2: axis('Detected-target RMS (mm)', {anchor: 'x2', side: 'right', rangemode: 'tozero'}),
        annotations: [{xref: 'paper', yref: 'paper', x: .18, y: 1.12, showarrow: false, text: 'Predicted power'},
          {xref: 'paper', yref: 'paper', x: .84, y: 1.12, showarrow: false, text: 'Measured error'}]}))
    ]);
  }

  function renderTargetSummary() {
    const short = {baseline: 'Baseline', reflectance_dark: 'Diffuse ρ = 0.02', reflectance_bright: 'Diffuse ρ = 0.85',
      target_yaw_45: 'Yaw 45°', target_yaw_75: 'Yaw 75°', target_short: 'Height 0.3 m',
      target_tall: 'Height 2.0 m', target_near: 'Target x = 9 m', target_far: 'Target x = 13 m'};
    const rows = DATA.targets.cases, labels = rows.map(row => short[row.id]);
    const specs = [['Eligible power (mW)', row => row.replay.mean_eligible_power_w * 1000, C.blue],
      ['Detected target returns', row => row.replay.detected_returns, C.teal],
      ['Conditional RMS (mm)', row => row.replay.range_rmse_m * 1000, C.orange]];
    const traces = [], axes = {}, annotations = [];
    specs.forEach((spec, index) => {
      const suffix = index ? String(index + 1) : '', domain = [[0, .25], [.365, .615], [.75, 1]][index];
      traces.push({x: rows.map(spec[1]), y: labels, type: 'scatter', mode: 'markers', marker: {color: spec[2], size: 8},
        xaxis: 'x' + suffix, yaxis: 'y' + suffix, showlegend: false,
        customdata: rows.map(row => [row.replay.eligible_echoes, row.replay.observed_detection_probability * 100]),
        hovertemplate: '%{y}<br>%{x:.5g}<br>%{customdata[0]:,} eligible; %{customdata[1]:.2f}% detected<extra>' + spec[0] + '</extra>'});
      axes['xaxis' + suffix] = axis(spec[0], {domain, anchor: 'y' + suffix, ...(index === 2 ? {range: [9.8, 10.3]} : {rangemode: 'tozero'})});
      axes['yaxis' + suffix] = axis('', {anchor: 'x' + suffix, categoryorder: 'array', categoryarray: [...labels].reverse(), showticklabels: index === 0});
    });
    return plot('target-summary', traces, layout('', '', {...axes, annotations, height: 370,
      margin: {l: 136, r: 15, t: 24, b: 62}}));
  }

  function renderWeatherRows() {
    const host = byId('weather-rows'); if (!host) return;
    host.replaceChildren(...DATA.primary.cases.map(row => {
      const a = row.aggregate, r = row.replay;
      const values = [row.label, format(a.mean_returns_per_control_step, 1), percent(a.mean_observed_fraction),
        format(1000 * r.range_rmse_m, 3), format(1000 * r.mean_eligible_power_w, 4),
        percent(r.observed_detection_probability), format(a.mean_elapsed_s, 2),
        format(a.mean_path_length_m, 3), format(a.worst_surface_gap_m, 3), a.goals + '/' + a.runs];
      const tr = document.createElement('tr');
      for (const value of values) {const td = document.createElement('td'); td.textContent = value; tr.append(td);}
      return tr;
    }));
  }

  function renderWeatherSummary() {
    const specs = [
      ['Mean points / scan', row => row.aggregate.mean_returns_per_control_step, 'Points / scan'],
      ['Observed map cells', row => row.aggregate.mean_observed_fraction * 100, 'Observed ROI cells (%)'],
      ['Conditional target RMS', row => row.replay.range_rmse_m * 1000, 'Detected-target RMS (mm)']
    ];
    const traces = [], axes = {}, annotations = [];
    specs.forEach((spec, index) => {
      const suffix = index ? String(index + 1) : '', domain = [[0, .27], [.365, .635], [.73, 1]][index];
      traces.push({type: 'bar', x: weatherIds.map(id => weatherNames[id]),
        y: weatherIds.map(id => spec[1](getCase(id))), marker: {color: weatherIds.map(id => weatherColors[id])},
        showlegend: false, xaxis: 'x' + suffix, yaxis: 'y' + suffix,
        hovertemplate: '%{x}<br>%{y:.3f}<extra>' + spec[2] + '</extra>'});
      axes['xaxis' + suffix] = axis('', {domain, anchor: 'y' + suffix, tickangle: -20});
      axes['yaxis' + suffix] = axis(spec[2], {anchor: 'x' + suffix, rangemode: 'tozero'});
      annotations.push({xref: 'paper', yref: 'paper', x: (domain[0] + domain[1]) / 2, y: 1.16,
        text: spec[0], showarrow: false, xanchor: 'center', font: {size: 13}});
    });
    return plot('weather-summary', traces, layout('', '', {...axes, annotations,
      margin: {l: 62, r: 16, t: 42, b: 82}, height: 350}));
  }

  const profileSpecs = {
    returns: {source: 'runs', x: 'control_times', y: 'returns', scale: 1, title: 'Detected points / scan', ylabel: 'All measured points / scan'},
    coverage: {source: 'runs', x: 'control_times', y: 'observed_fraction', scale: 100, title: 'Observed map-cell accumulation', ylabel: 'Endpoint-observed ROI cells (%)'},
    rms: {source: 'replays', x: 'times', y: 'range_rmse_m', scale: 1000, title: 'Matched-pose target range RMS', ylabel: 'Detected-target RMS (mm)'},
    power: {source: 'replays', x: 'times', y: 'eligible_power_w', scale: 1000, title: 'Matched-pose eligible echo power', ylabel: 'Mean eligible power (mW)'},
    detection: {source: 'replays', x: 'times', y: 'detection_probability', scale: 100, title: 'Matched-pose target detection', ylabel: 'Detected / eligible target echoes (%)'},
    gap: {source: 'runs', x: 'control_times', y: 'swept_gap_m', scale: 1, title: 'Closed-loop swept clearance', ylabel: 'Robot–surface gap (m)'},
    speed: {source: 'runs', x: 'control_times', y: 'speed_mps', scale: 1, title: 'Closed-loop commanded speed', ylabel: 'Commanded speed (m/s)'},
    route: {source: 'runs', x: 'times', y: 'route_error_m', scale: 1, title: 'Closed-loop reference-route error', ylabel: 'Distance to reference path (m)'},
    path: {source: 'runs', x: 'times', y: 'path_m', scale: 1, title: 'Closed-loop cumulative path length', ylabel: 'Travelled path length (m)'}
  };
  function renderProfile(id, key, withLegend = true) {
    const spec = profileSpecs[key];
    const traces = DATA.weather[spec.source].map(row => ({x: row[spec.x],
      y: row[spec.y].map(value => finite(value) ? value * spec.scale : null), type: 'scatter', mode: 'lines',
      name: row.label + ' · seed ' + row.seed, legendgroup: row.case_id, showlegend: withLegend && row.seed === 1,
      line: {color: weatherColors[row.case_id], width: row.seed === 1 ? 2.3 : 1.3,
        dash: row.seed === 1 ? 'solid' : row.seed === 2 ? 'dash' : 'dot'}, opacity: row.seed === 1 ? 1 : .72,
      connectgaps: false, hovertemplate: 'Recorded t=%{x:.1f} s<br>%{y:.4g}<extra>%{fullData.name}</extra>'}));
    return plot(id, traces, layout(spec.source === 'replays' ? 'Baseline replay time (s)' : 'Driven-run time (s)', spec.ylabel, {
      title: {text: spec.title, x: .02, xanchor: 'left', font: {size: 14}}, height: withLegend ? 390 : 320,
      margin: {l: 64, r: 16, t: withLegend ? 105 : 40, b: 58},
      legend: {orientation: 'h', x: 0, y: 1.28, font: {size: 10}},
      yaxis: axis(spec.ylabel, {rangemode: 'tozero'})}));
  }

  function obstacleTrace(object, axisIndex) {
    let points;
    if (object.type === 'plane') return null;
    if (object.type === 'box') {
      const angle = (object.yaw_deg || 0) * Math.PI / 180, c = Math.cos(angle), s = Math.sin(angle);
      points = [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]].map(([x, y]) => {
        const dx = x * object.size[0] / 2, dy = y * object.size[1] / 2;
        return [object.center[0] + c * dx - s * dy, object.center[1] + s * dx + c * dy];
      });
    } else {
      const center = object.base || object.center;
      points = Array.from({length: 33}, (_, k) => [center[0] + object.radius * Math.cos(k * Math.PI / 16),
        center[1] + object.radius * Math.sin(k * Math.PI / 16)]);
    }
    const suffix = axisIndex ? String(axisIndex + 1) : '';
    return {x: points.map(p => p[0]), y: points.map(p => p[1]), xaxis: 'x' + suffix, yaxis: 'y' + suffix,
      type: 'scatter', mode: 'lines', fill: 'toself', fillcolor: object.label === 'study_target' ? 'rgba(200,117,63,.25)' : 'rgba(97,114,123,.17)',
      line: {color: '#8D9AA1', width: .8}, showlegend: false, name: object.label,
      hovertemplate: '%{fullData.name}<extra>Unchanged scene geometry</extra>'};
  }
  function renderTrajectories() {
    const traces = [], axes = {}, annotations = [];
    weatherIds.forEach((caseId, index) => {
      const suffix = index ? String(index + 1) : '', domain = [[0, .29], [.355, .645], [.71, 1]][index];
      DATA.weather.scene_objects.forEach(object => {const trace = obstacleTrace(object, index); if (trace) traces.push(trace);});
      const reference = DATA.weather.runs.find(row => row.case_id === caseId).reference_xy;
      traces.push({x: reference.map(p => p[0]), y: reference.map(p => p[1]), xaxis: 'x' + suffix, yaxis: 'y' + suffix,
        mode: 'lines', line: {color: '#9CA7AE', width: 1.4, dash: 'dash'}, name: 'Fixed planning reference',
        showlegend: index === 0, hovertemplate: 'Fixed A* reference<extra></extra>'});
      for (const row of DATA.weather.runs.filter(run => run.case_id === caseId)) {
        traces.push({x: row.xy.map(p => p[0]), y: row.xy.map(p => p[1]), xaxis: 'x' + suffix, yaxis: 'y' + suffix,
          mode: 'lines', name: 'Seed ' + row.seed, legendgroup: 'seed-' + row.seed, showlegend: index === 0,
          line: {color: weatherColors[caseId], width: row.seed === 1 ? 2.4 : 1.5,
            dash: row.seed === 1 ? 'solid' : row.seed === 2 ? 'dash' : 'dot'},
          customdata: row.times, hovertemplate: 'x=%{x:.3f} m, y=%{y:.3f} m<br>t=%{customdata:.1f} s<extra>' + row.label + ' · seed ' + row.seed + '</extra>'});
      }
      traces.push({x: [0, 18], y: [0, 0], xaxis: 'x' + suffix, yaxis: 'y' + suffix,
        mode: 'markers+text', marker: {size: 7, color: [C.ink, C.orange], symbol: ['circle', 'star']},
        text: ['Start', 'Goal'], textposition: ['top right', 'top left'], showlegend: false, hoverinfo: 'skip'});
      axes['xaxis' + suffix] = axis('x (m)', {domain, anchor: 'y' + suffix, range: [-2, 21]});
      axes['yaxis' + suffix] = axis(index ? '' : 'y (m)', {anchor: 'x' + suffix, range: [-6, 7],
        scaleanchor: 'x' + suffix, scaleratio: 1});
      annotations.push({xref: 'paper', yref: 'paper', x: (domain[0] + domain[1]) / 2, y: 1.05,
        text: weatherNames[caseId], showarrow: false, xanchor: 'center', font: {size: 14}});
    });
    return plot('weather-trajectories', traces, layout('', '', {...axes, annotations, height: 440,
      margin: {l: 46, r: 15, t: 80, b: 45},
      legend: {orientation: 'h', x: 0, y: -.14, font: {size: 10}}}));
  }

  async function init(data) {
    DATA = data;
    optionList('dependency-case', data.targets.cases.map(row => [row.id, row.label]), 'reflectance_dark');
    optionList('threshold-pose', [['all', 'All three viewpoints'], ...data.threshold.views.map(row => [row.pose_id, row.label])], 'all');
    optionList('weather-extra', Object.entries(profileSpecs).map(([key, spec]) => [key, spec.title]), 'gap');
    const snr = byId('threshold-snr');
    if (snr) {snr.min = 0; snr.max = 10; snr.step = 1; snr.value = 5;}
    for (const id of ['power-range', 'power-rho', 'power-kappa', 'power-angle']) byId(id)?.addEventListener('input', renderPower);
    byId('dependency-case')?.addEventListener('change', renderDependencies);
    byId('threshold-pose')?.addEventListener('change', renderThreshold);
    byId('threshold-snr')?.addEventListener('input', renderThreshold);
    byId('weather-extra')?.addEventListener('change', () => renderProfile('weather-extra-plot', byId('weather-extra').value, true));
    renderWeatherRows();
    await Promise.all([renderPower(), renderDependencies(), renderThreshold(), renderWeatherSummary(), renderTargetSummary(), renderTrajectories(),
      renderProfile('profile-returns', 'returns'), renderProfile('profile-coverage', 'coverage', false),
      renderProfile('profile-rms', 'rms', false), renderProfile('profile-power', 'power', false),
      renderProfile('weather-extra-plot', 'gap')]);
  }
  function resize() {
    for (const id of rendered) {
      const host = byId(id);
      if (host && host.clientWidth && host.getClientRects().length) Plotly.Plots.resize(host);
    }
  }
  window.ReportCharts = {init, resize};
}());
