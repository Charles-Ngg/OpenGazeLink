/* Isolated browser regression: physical full-screen coordinates and fusion only. */
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {chromium} = require('playwright');
(async () => {
  const output = 'data/ui-checks/preview';
  fs.mkdirSync(output, {recursive:true});
  const config = JSON.parse(require('child_process').execFileSync(process.env.OPENGAZELINK_TEST_PYTHON || (process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python'),
    ['-c', 'import json; from dataclasses import asdict; from opengazelink_pc.config import ProviderConfig; print(json.dumps(asdict(ProviderConfig())))'], {encoding:'utf8'}));
  Object.assign(config, {screen_width:3840, screen_height:2160, geometry_configured:true});
  const status = {config, engine:{tracking:true,input:{ready:true}}, geometry:{configured:true},
    artifacts:{models:{tasks_conditioned_video:{ready:true,compatible:true}}},calibration:{active:false}};
  const browser = await chromium.launch({headless:true,channel:process.env.PLAYWRIGHT_CHANNEL || undefined});
  const results = [], errors = [];
  try {
    for (const viewport of [{width:1920,height:1080},{width:1280,height:720},{width:640,height:400}]) {
      const page = await browser.newPage({viewport});
      page.on('pageerror', e => errors.push(e.message));
      await page.addInitScript(() => {window.EventSource = class {close() {}};});
      await page.route('http://127.0.0.1:8879/**', async route => {
        const pathname = new URL(route.request().url()).pathname;
        if (pathname.startsWith('/api/')) return route.fulfill({json:pathname === '/api/status' ? status : {ok:true}});
        const file = path.join(process.cwd(), 'web', pathname === '/' ? 'index.html' : pathname.slice(1));
        if (!fs.existsSync(file)) return route.fulfill({status:404,body:''});
        await route.fulfill({body:fs.readFileSync(file),contentType:pathname.endsWith('.js') ? 'application/javascript' : pathname.endsWith('.css') ? 'text/css' : 'text/html'});
      });
      await page.goto('http://127.0.0.1:8879');
      await page.locator('[data-page="preview"]').click();
      await page.locator('#previewStartButton').click();
      await page.waitForFunction(() => document.fullscreenElement?.id === 'previewPage');
      const check = await page.evaluate(() => {
        const rect = document.getElementById('gazeStage').getBoundingClientRect();
        const points = [];
        for (const [x,y] of [[.01,.01],[.99,.01],[.5,.5],[.01,.99],[.99,.99]]) {
          renderGaze({valid:true,combined:[x*3839,y*2159],left:[0,0],right:[3839,2159]});
          const dot = document.getElementById('combinedDot').getBoundingClientRect();
          points.push({x:dot.x+dot.width/2,y:dot.y+dot.height/2,expectedX:x*innerWidth,expectedY:y*innerHeight});
        }
        return {rect:{x:rect.x,y:rect.y,width:rect.width,height:rect.height},width:innerWidth,height:innerHeight,points,
          dots:document.querySelectorAll('.gaze-dot').length,
          left:!!document.getElementById('leftDot'),right:!!document.getElementById('rightDot')};
      });
      assert.deepEqual(check.rect,{x:0,y:0,width:check.width,height:check.height});
      assert.equal(check.dots,1); assert.equal(check.left,false); assert.equal(check.right,false);
      for (const p of check.points) {assert(Math.abs(p.x-p.expectedX)<.05);assert(Math.abs(p.y-p.expectedY)<.05);}
      // A wrapping/multi-line toolbar must never resize or translate the screen.
      await page.evaluate(() => {document.getElementById('previewState').textContent='Status '.repeat(40);});
      const changed = await page.locator('#gazeStage').boundingBox();
      assert.deepEqual(changed,check.rect);
      await page.evaluate(() => renderGaze({valid:true,combined:[1919.5,21.59]}));
      await page.screenshot({path:`${output}/fullscreen-${viewport.width}.png`});
      await page.evaluate(() => renderGaze({valid:false}));
      assert(await page.locator('#combinedDot').isHidden());
      await page.evaluate(() => document.exitFullscreen());
      await page.waitForFunction(() => !document.fullscreenElement);
      const normal = await page.locator('#gazeStage').boundingBox();
      assert(normal.y>0);
      results.push({viewport,...check});
      await page.close();
    }
    assert.deepEqual(errors,[]);
    fs.writeFileSync(`${output}/verification.json`,JSON.stringify({results,errors},null,2));
    console.log('Full-screen viewport mapping, toolbar reflow, fusion-only display, invalid gaze and exit passed in 3 viewport sizes.');
  } finally {await browser.close();}
})().catch(e => {console.error(e);process.exitCode=1;});
