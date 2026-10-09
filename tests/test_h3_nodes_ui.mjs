import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

async function extension(name) {
    const source = (await readFile(new URL(`../web/${name}.js`, import.meta.url), 'utf8'))
        .replace('import { app } from "../../scripts/app.js";', 'const app = {registerExtension() {}};');
    return import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
}
const { migrateH3Workflow, unifiedDefaults } = await extension('h3_migration');
const { prepareLoRA } = await extension('h3_lora');
const node = (type, values) => ({id: 7, type, inputs: [{name:'model',link:3}], widgets_values: values});
const settings = (n) => Object.fromEntries(Object.keys(unifiedDefaults).map((name,i) => [name,n.widgets_values[i]]));

test('Turbo and ordinary single LoRA migrate weights, strictness, and mode', () => {
    for (const type of ['AppleSiliconH3TurboLoRA','AppleSiliconH3LoRA']) {
        const n = node(type,['saved.safetensors',-0.3,'merged']);
        migrateH3Workflow({nodes:[n]});
        assert.equal(n.type,'AppleSiliconH3LoRA');
        assert.deepEqual(JSON.parse(n.widgets_values[0]),[{enabled:true,lora_name:'saved.safetensors',strength:-0.3,
            validation:type.includes('Turbo')?'turbo':'standard'}]);
        assert.equal(n.widgets_values[1],'merged');
        assert.equal(n.inputs[0].link,3);
        const saved = structuredClone(n);
        migrateH3Workflow({nodes:[n]});
        assert.deepEqual(n,saved);
    }
});

test('Standalone sparse preserves every supported setting and disables other groups', () => {
    const n = node('AppleSiliconH3SparseAttention',['corrected',0.7,0.5,3,8192,false,0.91,0,64]);
    migrateH3Workflow({nodes:[n]});
    const out = settings(n);
    assert.equal(n.type,'AppleSiliconH3UnifiedAcceleration');
    assert.deepEqual([out.sparse_mode,out.sparse_tau,out.sparse_start_percent,out.sparse_dense_leading_blocks,
        out.sparse_min_sequence_tokens,out.sparse_verbose,out.sparse_end_percent],['corrected',0.7,0.5,3,8192,false,0.91]);
    assert.equal(out.cache_mode,'disabled'); assert.equal(out.token_reduction,false); assert.equal(out.sparse_attention,true);
});

test('Cache and token reduction migration preserve numerical controls', () => {
    const c=node('AppleSiliconH3Cache',[0.13,0.25,0.88,3,0.8,0,'first_block_cache_per_step',false]);
    const t=node('AppleSiliconH3StructuralAcceleration',[true,50,0.9,1,12,32,42,true]);
    migrateH3Workflow({nodes:[c,t]});
    assert.equal(settings(c).cache_mode,'first_block_cache_per_step');
    assert.equal(settings(c).cache_reuse_threshold,0.13); assert.equal(settings(c).cache_verbose,false);
    assert.equal(settings(c).sparse_attention,false);
    assert.equal(settings(t).token_reduction_scale,0.9); assert.equal(settings(t).token_early_evaluations,12);
    assert.equal(settings(t).token_detail_restore_block,32); assert.equal(settings(t).token_coarse_restore_block,42);
    assert.equal(settings(t).token_verbose,true); assert.equal(settings(t).cache_mode,'disabled');
});

test('Uncalibrated presets, failed research modes and connected widgets are never silently changed', () => {
    for (const n of [node('AppleSiliconH3Acceleration',['balanced',true]),
        node('AppleSiliconH3Cache',[0.2,0.2,0.85,1,1,0,'velocity_hold',true]),
        node('AppleSiliconH3SparseAttention',['hard_drop',1,0.5,2,8192,true]),
        node('AppleSiliconH3SparseAttention',['corrected',1,0.5,2,8192,true,0.85,2,32])]) {
        const original=structuredClone(n); const warnings=[];
        migrateH3Workflow({nodes:[n]}, m=>warnings.push(m));
        assert.deepEqual(n,original); assert.equal(warnings.length,1);
    }
    const n=node('AppleSiliconH3TurboLoRA',['a',1,'fast_runtime']);n.inputs.push({name:'strength',link:9});
    const original=structuredClone(n); migrateH3Workflow({nodes:[n]},()=>{}); assert.deepEqual(n,original);
});

test('Exact placeholder migrates to a no-op, including in subgraphs', () => {
    const n=node('AppleSiliconH3Acceleration',['exact',true]);
    migrateH3Workflow({nodes:[],definitions:{subgraphs:[{nodes:[n]}]}});
    assert.deepEqual(settings(n), unifiedDefaults);
});

test('Dynamic list supports add, disable, signed weight, remove, restore, and API serialization', () => {
    const n={widgets:[{name:'loras',value:'[]'},{name:'mode',value:'fast_runtime'}],size:[390,100],
        addWidget(type,name,value,callback,options){const w={type,name,value,callback,options};this.widgets.push(w);return w;},
        setSize(size){this.size=size;},computeSize(){return [390,this.widgets.length*25];}};
    prepareLoRA(n,['a.safetensors','b.safetensors']);
    const w=(name)=>n.widgets.find(w=>w.name===name);
    w('Add LoRA').callback(); w('Add LoRA').callback();
    w('LoRA 1 enabled').callback(false);w('LoRA 2 file').callback('b.safetensors');
    w('LoRA 2 weight').callback(-0.45);w('LoRA 2 validation').callback('turbo');
    const saved=w('loras').serializeValue();
    const info={};n.onSerialize(info);assert.deepEqual(info.widgets_values,[saved,'fast_runtime']);
    assert.deepEqual(JSON.parse(saved),[
        {enabled:false,lora_name:'a.safetensors',strength:1,validation:'standard'},
        {enabled:true,lora_name:'b.safetensors',strength:-0.45,validation:'turbo'}]);
    assert.deepEqual(n.widgets.filter(w=>w.options?.serialize!==false).map(w=>w.name),['loras','mode']);
    n.onConfigure();assert.equal(w('LoRA 1 enabled').value,false);assert.equal(w('LoRA 2 weight').value,-0.45);
    w('Remove LoRA 1').callback(); assert.equal(JSON.parse(w('loras').value).length,1);
    assert.equal(w('LoRA 1 file').value,'b.safetensors');
    w('Remove LoRA 1').callback();assert.equal(w('loras').value,'[]');
});

async function captureExtension(name) {
    const source = (await readFile(new URL(`../web/${name}.js`, import.meta.url), 'utf8'))
        .replace('import { app } from "../../scripts/app.js";', '');
    let registered;
    new Function('app', source)({registerExtension(value) { registered = value; }});
    return registered;
}

test('Latent upscaler shows only the selected size controls and preserves saved values', async () => {
    const ext = await captureExtension('h3_latent_upscale');
    class Node {
        constructor() {
            this.widgets = [
                {name:'mode',value:'scale',type:'combo'},
                {name:'scale',value:2,type:'number'},
                {name:'target_width',value:1728,type:'number'},
                {name:'target_height',value:960,type:'number'},
                {name:'megapixels',value:1,type:'number'},
            ];
        }
        computeSize() { return [400,200]; }
        setSize(value) { this.size = value; }
    }
    ext.beforeRegisterNodeDef(Node, {name:'AppleSiliconH3LatentUpscale'});
    const n = new Node();n.onNodeCreated();
    const w = name => n.widgets.find(w=>w.name===name);
    assert.equal(w('scale').type,'number');
    assert.match(w('target_width').type,/hidden/);
    w('mode').value='dimensions';w('mode').callback();
    assert.match(w('scale').type,/hidden/);
    assert.equal(w('target_width').type,'number');
    assert.equal(w('target_width').value,1728);
    w('mode').value='megapixels';n.onConfigure();
    assert.equal(w('megapixels').type,'number');
    assert.match(w('target_height').type,/hidden/);
    assert.equal(n.widgets.length,5);
});




async function controlsNode(saved) {
    const source=(await readFile(new URL('../web/h3_acceleration.js',import.meta.url),'utf8'))
        .replace('import { app } from "../../scripts/app.js";','const app={registerExtension(){}};')
        .replace('import { buildH3Panel } from "./h3_panel.js";','');
    const {accelerationControls,PRESETS}=await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
    const defaults={...unifiedDefaults,cache_mode:'motion_cache',sparse_backend:'bf16',sparse_dense_layers:'',sparse_window_mode:'sigma_percent',
        motion_reuse_threshold:0.15,motion_strength:1,motion_warmup_steps:4,motion_max_consecutive_skips:2,
        motion_start_percent:0.15,motion_end_percent:0.95,spectrum_blend_weight:0.5,sparse_memory_budget_gib:0,
        veda_predictor:'predictor.safetensors',veda_sparsity:90,veda_reference_sparsity:0,veda_dense_steps:'',sparse_topk_ratio:0,token_start_percent:0,token_end_percent:1};
    const n={widgets:Object.entries({...defaults,...saved}).map(([name,value])=>({name,value})),properties:{}};
    return {n,c:accelerationControls(n),presets:PRESETS};
}

test('Default is MotionCache alone; sparse opt-in starts with the measured SOL recipe',async()=>{
    const {c}=await controlsNode();
    assert.equal(c.selectedPreset(),'MotionCache · recommended');
    c.set('sparse_attention',true);
    assert.equal(c.get('cache_mode'),'motion_cache');
    assert.equal(c.get('sparse_backend'),'int8');assert.equal(c.get('sparse_tau'),1.3);
    assert.equal(c.get('sparse_window_mode'),'step_fraction');assert.equal(c.get('sparse_min_sequence_tokens'),8192);
    assert.equal(c.selectedPreset(),'Custom combination');
    c.set('sparse_tau',0.45);c.set('sparse_attention',false);c.set('sparse_attention',true);
    assert.equal(c.get('sparse_tau'),0.45);
});

test('Veda is opt-in and switches independently while retaining SOL and Veda tuning',async()=>{
    const {c}=await controlsNode();
    c.preset('Veda · protected (experimental)');
    assert.equal(c.get('sparse_backend'),'veda_metal');
    assert.equal(c.get('cache_mode'),'disabled');
    assert.equal(c.get('veda_reference_sparsity'),0);
    assert.equal(c.selectedPreset(),'Veda · protected (experimental)');
    assert.equal(c.get('sparse_min_sequence_tokens'),8192);
    c.preset('Veda · fast (experimental)');
    assert.equal(c.get('sparse_dense_layers'),'');
    assert.equal(c.get('sparse_start_percent'),0);
    assert.equal(c.get('sparse_end_percent'),1);
    assert.equal(c.get('cache_mode'),'disabled');
    c.set('veda_sparsity',80);c.set('sparse_tau',0.45);
    c.set('sparse_backend','int8');c.set('sparse_backend','veda_mlx');
    assert.equal(c.get('veda_sparsity'),80);assert.equal(c.get('sparse_tau'),0.45);
    c.set('cache_mode','motion_cache');
    assert.equal(c.get('sparse_attention'),true);
    c.preset('Exact · no approximation');
    assert.equal(c.get('sparse_attention'),false);
    assert.equal(c.get('cache_mode'),'disabled');
});

test('Cache choice, sparse, and token switches are independent and retain tuning',async()=>{
    const {c}=await controlsNode();c.set('sparse_attention',true);c.set('token_reduction',true);
    c.set('cache_mode','spectrum_protected');c.set('spectrum_blend_weight',0.35);
    assert.equal(c.get('sparse_attention'),true);assert.equal(c.get('token_reduction'),true);
    c.set('cache_mode','disabled');assert.equal(c.get('sparse_attention'),true);
    c.set('cache_mode','spectrum_protected');assert.equal(c.get('spectrum_blend_weight'),0.35);
    c.preset('Exact · no approximation');
    assert.equal(c.get('cache_mode'),'disabled');assert.equal(c.get('sparse_attention'),false);assert.equal(c.get('token_reduction'),false);
});

test('Loaded custom sparse configuration is never reset when enabled',async()=>{
    const {n,c}=await controlsNode({sparse_tau:0.123,sparse_backend:'bf16',sparse_window_mode:'sigma_percent'});
    n.properties.asfp8_h3_sparse_configured=true;
    c.set('sparse_attention',true);assert.equal(c.get('sparse_tau'),0.123);
    assert.equal(c.get('sparse_backend'),'bf16');assert.equal(c.get('sparse_window_mode'),'sigma_percent');
});

test('Connected settings cannot be replaced by presets or local controls',async()=>{
    const {n,c}=await controlsNode();n.inputs=[{widget:{name:'cache_mode'},link:7}];
    const original=n.widgets.map(w=>w.value);c.preset('Exact · no approximation');c.set('cache_mode','disabled');
    assert.deepEqual(n.widgets.map(w=>w.value),original);assert.equal(c.selectedPreset(),'Custom combination');
    c.set('token_reduction',true);assert.equal(c.get('token_reduction'),true);
});

test('SOL memory ceiling survives recipe changes and independent toggles',async()=>{
    const {n,c}=await controlsNode();assert.equal(c.get('sparse_memory_budget_gib'),0);
    c.set('sparse_memory_budget_gib',16);
    for (const preset of ['SOL attention · 14 steps','Exact · no approximation','SOL + FirstBlockCache']) {
        c.preset(preset);assert.equal(c.get('sparse_memory_budget_gib'),16);
        assert.equal(c.selectedPreset(),preset);
    }
    c.set('sparse_attention',false);c.set('sparse_attention',true);
    assert.equal(c.get('sparse_memory_budget_gib'),16);
    assert.equal(n.widgets.findIndex(w=>w.name==='sparse_memory_budget_gib'),n.widgets.length-8);
});

test('Every approximate preset agrees with its measured benchmark settings',async()=>{
    const fixtures={'MotionCache · recommended':'h3_motion_cache','Spectrum · protected':'h3_spectrum_protected',
        'Spectrum · fast draft':'h3_spectrum_fast','FirstBlockCache · 0.20':'h3_first_block_cache',
        'SOL attention · 14 steps':'h3_sol14','SOL + FirstBlockCache':'h3_sol14_cache'};
    for(const [label,file] of Object.entries(fixtures)) {
        const graph=JSON.parse(await readFile(new URL(`../workflows/${file}.api.json`,import.meta.url),'utf8'));
        const reference=Object.values(graph).find(n=>n.class_type==='AppleSiliconH3UnifiedAcceleration').inputs;
        const {c}=await controlsNode();c.preset(label);
        for(const name of Object.keys(reference).filter(k=>['cache_mode','token_reduction','sparse_attention'].includes(k)||
            (k.startsWith('cache_')&&reference.cache_mode==='first_block_cache')||
            (k.startsWith('sparse_')&&reference.sparse_attention))) assert.equal(c.get(name),reference[name],`${label}: ${name}`);
        assert.equal(c.selectedPreset(),label);
    }
});


test('New sparse budgets reset on old recipes, and token protection is independent',async()=>{
    const {c}=await controlsNode();
    c.preset('SOL top-k · wide');assert.equal(c.get('sparse_topk_ratio'),.15);
    assert.equal(c.get('sparse_start_percent'),.10);
    c.preset('SOL top-k · all-step draft');assert.equal(c.get('sparse_topk_ratio'),.10);
    assert.equal(c.get('sparse_dense_layers'),'');
    c.preset('SOL attention · 14 steps');assert.equal(c.get('sparse_topk_ratio'),0);
    c.preset('Token reduction · protected');assert.equal(c.get('token_start_percent'),.20);
    assert.equal(c.get('token_end_percent'),.90);assert.equal(c.get('token_coarse_restore_block'),12);
    c.set('sparse_attention',true);assert.equal(c.get('token_start_percent'),.20);
    c.set('token_reduction',false);c.set('token_reduction',true);
    assert.equal(c.get('token_end_percent'),.90);
});

test('Dense and hybrid recipes remain independent of cache and token controls',async()=>{
    const {c}=await controlsNode();
    c.preset('Dense INT8 · protected');
    assert.equal(c.get('sparse_backend'),'dense_int8');
    assert.equal(c.get('sparse_start_percent'),.20);
    assert.equal(c.get('sparse_end_percent'),.90);
    assert.equal(c.get('cache_mode'),'disabled');
    c.set('cache_mode','motion_cache');c.set('token_reduction',true);
    assert.equal(c.get('sparse_backend'),'dense_int8');
    c.preset('Dense INT8 · full draft');
    assert.equal(c.get('sparse_dense_layers'),'');
    assert.equal(c.get('sparse_dense_leading_blocks'),0);
    assert.equal(c.get('sparse_start_percent'),0);
    assert.equal(c.get('sparse_end_percent'),1);
    c.preset('SOL INT8 · hybrid fast');
    assert.equal(c.get('sparse_backend'),'int8_hybrid');
    assert.equal(c.get('sparse_tau'),1.3);
    assert.equal(c.get('sparse_dense_layers'),'0,1,33,34,35');
    c.preset('SOL attention · 14 steps');
    assert.equal(c.get('sparse_backend'),'int8');
});
