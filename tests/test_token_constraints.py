"""The model selects tokens; the grammar cannot fabricate a hybrid config."""
import json
import pytest
from kernel_tuning_lab.token_constraints import ConfigTokenTrie
from kernel_tuning_lab.models import KernelConfig,canonical_json


class CharacterTokenizer:
    def __call__(self,texts,**kwargs):return {'input_ids':[[ord(c) for c in text] for text in texts]}


def enumerate_outputs(trie,prefix=()):
    allowed=trie.allowed(prefix,[0])
    if allowed==[0]:return [''.join(map(chr,prefix))]
    return [text for token in allowed for text in enumerate_outputs(trie,(*prefix,token))]


def test_disconnected_config_regions_stay_exact_without_hybrid_or_random_pool():
    configs=[KernelConfig({'x':1,'y':9}),KernelConfig({'x':8,'y':2})]
    trie=ConfigTokenTrie(configs,CharacterTokenizer())
    assert set(enumerate_outputs(trie))=={canonical_json({'config':dict(c.params)}) for c in configs}
    trie.exclude([configs[0].id]);trie.exclude([configs[0].id])
    assert trie.remaining==1 and json.loads(enumerate_outputs(trie)[0])['config']=={'x':8,'y':2}
    trie.exclude([configs[1].id]);assert trie.remaining==0
    with pytest.raises(ValueError):trie.allowed([], [0])


def test_grammar_reaches_a_point_outside_any_old_512_pool_and_enforces_build_bound():
    configs=[KernelConfig({'x':n}) for n in range(600)]
    trie=ConfigTokenTrie(configs,CharacterTokenizer());trie.exclude(c.id for c in configs[:-1])
    assert json.loads(enumerate_outputs(trie)[0])['config']=={'x':599}
    with pytest.raises(ValueError,match='bounded node'):ConfigTokenTrie(configs,CharacterTokenizer(),max_nodes=4)
