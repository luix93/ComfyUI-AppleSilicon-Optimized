"""Release builds retain upstream and transitive dependency notices."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sr_notice_setup', ROOT / 'dev/setup_super_resolution.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def source_tree(tmp_path):
    source=tmp_path/'source';source.mkdir()
    for name in ('LICENSE','NOTICE'):(source/name).write_text('Runtime '+name)
    deps=source/'.build/checkouts'
    for name,license in [('mlx-swift','Swift MIT'),('swift-numerics','Numerics Apache'),('swift-argument-parser','Parser Apache')]:
        path=deps/name;path.mkdir(parents=True);(path/'LICENSE.txt').write_text(license)
    nested=deps/'mlx-swift/vendor/thirdparty';nested.mkdir(parents=True)
    (nested/'NOTICE').write_text('Nested dependency copyright')
    staged=tmp_path/'staged';staged.mkdir()
    return source,staged


def test_every_resolved_notice_is_preserved(tmp_path):
    source,staged=source_tree(tmp_path)
    setup.stage_runtime_notices(source,staged)
    assert (staged/'NOTICE-MLX-DLSS.txt').read_text()=='Runtime NOTICE'
    assert (staged/'LICENSE-MLX-DLSS.txt').read_text()=='Runtime LICENSE'
    for name in ('mlx-swift','swift-numerics','swift-argument-parser'):
        assert (staged/'THIRD_PARTY_LICENSES'/name/'LICENSE.txt').read_bytes()==(source/'.build/checkouts'/name/'LICENSE.txt').read_bytes()
    assert (staged/'THIRD_PARTY_LICENSES/mlx-swift/vendor/thirdparty/NOTICE').read_text()=='Nested dependency copyright'
    manifest=json.loads((staged/'DEPENDENCY-NOTICES.json').read_text())
    assert len(manifest['notices'])==5
    for entry in manifest['notices']:assert setup.hashlib.sha256((staged/entry['file']).read_bytes()).hexdigest()==entry['sha256']


def test_build_refuses_missing_dependency_notice(tmp_path):
    source,staged=source_tree(tmp_path)
    (source/'.build/checkouts/swift-argument-parser/LICENSE.txt').unlink()
    with pytest.raises(RuntimeError,match='No license/notice found.*swift-argument-parser'):
        setup.stage_runtime_notices(source,staged)


def test_build_refuses_missing_upstream_notice(tmp_path):
    source,staged=source_tree(tmp_path);(source/'NOTICE').unlink()
    with pytest.raises(RuntimeError,match='Missing upstream runtime notice'):
        setup.stage_runtime_notices(source,staged)


def test_external_symlink_is_not_copied_as_a_notice(tmp_path):
    source,staged=source_tree(tmp_path)
    target=tmp_path/'outside';target.write_text('outside checkout')
    (source/'.build/checkouts/swift-numerics/NOTICE.external').symlink_to(target)
    setup.stage_runtime_notices(source,staged)
    assert not (staged/'THIRD_PARTY_LICENSES/swift-numerics/NOTICE.external').exists()
