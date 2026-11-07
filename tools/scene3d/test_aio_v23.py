"""Offline tests only. No API prompt export, HTTP calls or GPU work."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('patch_aio_v23', HERE/'patch_aio_v23.py')
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


class V23Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_path = p.ROOT/'workflows/aio/AIO_manual_V2.2.ui.json'
        cls.raw = cls.source_path.read_bytes()
        cls.source = json.loads(cls.raw)
        cls.result = p.patch(cls.source)

    def test_links_slots_types_interfaces(self):
        self.assertEqual([], p.validate(self.result))

    def test_default_off_legacy_equivalence(self):
        self.assertTrue(p.off_compare(self.source,self.result))
        self.assertEqual(self.raw,self.source_path.read_bytes())

    def test_deterministic_and_saved_output(self):
        self.assertEqual(self.result,p.patch(self.source))
        saved=json.loads((p.ROOT/'workflows/aio/AIO_manual_V2.3.ui.json').read_text(encoding='utf-8'))
        self.assertEqual(self.result,saved)

    def test_existing_widget_literals_are_opaque_unchanged(self):
        originals={n['id']:n for g in p.graphs(self.source) for n in g['nodes']}
        for g in p.graphs(self.result):
            for n in g['nodes']:
                if n['id'] in originals:
                    old=originals[n['id']]
                    self.assertEqual(old.get('widgets_values'),n.get('widgets_values'))
                    self.assertEqual(old.get('mode'),n.get('mode'))
                    self.assertEqual(old.get('title'),n.get('title'))

    def test_reject_enabled_default_and_legacy_widget_drift(self):
        r=copy.deepcopy(self.result)
        ctl=r['extra']['aio_v23']['controls']['enabled']['title']
        next(n for n in r['nodes'] if n.get('title')==ctl)['widgets_values']=[True]
        with self.assertRaises(AssertionError):p.off_compare(self.source,r)
        r=copy.deepcopy(self.result)
        next(n for n in r['nodes'] if n['type']=='NGFarmPanelPhoto')['widgets_values'][0]=True
        with self.assertRaises(AssertionError):p.off_compare(self.source,r)

    def test_detect_reverse_endpoint_and_type_corruption(self):
        r=copy.deepcopy(self.result)
        s=next(g for g in p.graphs(r) if g.get('id')==r['extra']['aio_v23']['photo_id'])
        link=next(l for l in s['links'] if l['origin_id'] in r['extra']['aio_v23']['added_nodes'])
        link['type']='WRONG_TYPE'
        self.assertTrue(p.validate(r))
        r=copy.deepcopy(self.result)
        n=next(n for n in r['nodes'] if n.get('title')==r['extra']['aio_v23']['controls']['enabled']['title'])
        n['outputs'][0]['links'].append(r['links'][0][0])
        self.assertTrue(p.validate(r))

    def test_recipe_defaults_and_server_composite(self):
        cs=self.result['extra']['aio_v23']['controls']
        self.assertEqual((False,True,25),(cs['enabled']['default'],cs['video1']['default'],cs['steps']['default']))
        self.assertFalse({'pose','harmonize','denoise45','pose_strength','pose_end','folder'} & set(cs))
        nodes={n.get('title'):n for g in p.graphs(self.result) for n in g['nodes'] if n['id'] in self.result['extra']['aio_v23']['added_nodes']}
        types={n['type'] for n in nodes.values()}
        # server variant: no Fun CN, no patch loader, no harmonisation sampler, no turbo LoRA
        self.assertFalse(types & {'MiniMaxH3FunControlNetApply','ModelPatchLoader','LoadImagesFromFolderKJ','H3LatentFromImage','SamplerCustomAdvanced','LoraLoaderModelOnly'})
        self.assertEqual([5],nodes['3D · INPUT / playblast ×5 frames @24fps']['widgets_values'])
        self.assertEqual([0.8,'multiply'],nodes['3D · COMPOSITE / photo x (1 - 0.8 shadow)']['widgets_values'])
        self.assertEqual([6,0,True,False,9,1,1,False],nodes['3D · COMPOSITE / feather person']['widgets_values'])
        self.assertIn('3D · COMPOSITE / кадр → размер V2.2',nodes)
        s=next(g for g in p.graphs(self.result) if g.get('id')==self.result['extra']['aio_v23']['photo_id'])
        comp=nodes['3D · COMPOSITE / person onto photo']
        dest=next(i for i in comp['inputs'] if i['name']=='destination')['link']
        link=next(l for l in s['links'] if l['id']==dest)
        self.assertEqual(nodes['3D · COMPOSITE / photo x (1 - 0.8 shadow)']['id'],link['origin_id'])

    def test_new_nodes_against_targeted_object_info(self):
        schemas=json.loads((HERE/'aio_v23_object_info.json').read_text(encoding='utf-8'))
        added=set(self.result['extra']['aio_v23']['added_nodes'])
        for g in p.graphs(self.result):
            for n in g['nodes']:
                if n['id'] not in added or n['type']=='MarkdownNote':continue
                schema=schemas[n['type']]
                fields={key:value for sec in schema['input'].values() for key,value in sec.items()}
                sockets={i['name']:i for i in n.get('inputs',[])}
                for name in sockets:
                    self.assertTrue(name in fields or name.startswith(('ref_images.','ref_videos.','ref_audios.','ref_video_audios.','values.')),
                                    (n['title'],name))
                self.assertEqual(len(schema['output']),len(n['outputs']),n['title'])
                # Newly authored named widgets have a synchronized positional representation.
                named=n.get('widgets_values_named')
                if named:
                    self.assertEqual(list(named.values()),n['widgets_values'])
                    for key,value in named.items():
                        if key=='upload':continue  # frontend-only button state
                        self.assertIn(key,fields,n['title'])
                        spec=fields[key]; opts=spec[0] if isinstance(spec[0],list) else spec[1].get('options') if len(spec)>1 else None
                        if opts is not None and opts:self.assertIn(value,opts,(n['title'],key))
                for key,spec in schema['input'].get('required',{}).items():
                    typ=spec[0]
                    if isinstance(typ,str) and typ not in ('INT','FLOAT','BOOLEAN','STRING','COMBO','COLORCODE','COMFY_AUTOGROW_V3','COMFY_MATCHTYPE_V3'):
                        if key not in ('values',):
                            self.assertIsNotNone(sockets.get(key,{}).get('link'),(n['title'],key))

    def test_new_sfw_text_camera_light_color_and_fallback(self):
        for g in p.graphs(self.result):
            for n in g['nodes']:
                if n['id'] not in self.result['extra']['aio_v23']['added_nodes']:continue
                if n['type']!='PrimitiveStringMultiline':continue
                text=n['widgets_values'][0]
                self.assertNotIn('studio',text.lower())
                self.assertIn('static, locked-off',text)
                self.assertIn('\nThe lighting is preserved unchanged throughout.\n',text)
                self.assertIn('\nColors and saturation stay unchanged from the reference image.\n',text)
                if 'fallback' in n['title']:self.assertNotIn('<Video 1>',text)


if __name__=='__main__':unittest.main()
