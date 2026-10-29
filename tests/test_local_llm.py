"""Checkpoint identity tests use tiny fixtures, never claim model inference."""
import json

import pytest

from kernel_tuning_lab.local_llm import checkpoint_manifest,verify_manifest


def test_existing_checkpoint_complete_shards_and_tokenizer_are_fingerprinted(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'model_type':'fixture_not_real_LLM'}))
    (tmp_path/'tokenizer_config.json').write_text('{}')
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':'a.safetensors','y':'b.safetensors'}}))
    (tmp_path/'a.safetensors').write_bytes(b'fixture-a')
    with pytest.raises(ValueError,match='incomplete'):checkpoint_manifest(tmp_path,'unit')
    (tmp_path/'b.safetensors').write_bytes(b'fixture-b')
    result=checkpoint_manifest(tmp_path,'unit',hash_weights=True)
    assert result['full_weight_hashes'] and len(result['weights'])==2
    assert result['weights_bytes']==18 and not result['downloads'] and not result['training']
    verify_manifest(result)
    (tmp_path/'tokenizer_config.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='metadata/tokenizer changed'):verify_manifest(result)


def test_changed_weights_and_nonlocal_paths_are_rejected(tmp_path):
    (tmp_path/'config.json').write_text('{"model_type":"fixture"}')
    (tmp_path/'model.safetensors').write_bytes(b'not actual weights')
    result=checkpoint_manifest(tmp_path,'unit',hash_weights=True)
    (tmp_path/'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='weight changed'):verify_manifest(result)
    (tmp_path/'model.safetensors.index.json').write_text('{"weight_map":{"x":"../forbidden.safetensors"}}')
    with pytest.raises(ValueError,match='complete local safetensors'):checkpoint_manifest(tmp_path,'unit')
