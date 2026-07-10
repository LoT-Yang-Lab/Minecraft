# -*- coding: utf-8 -*-
"""项目环境初始化。

本模块负责设置 Python 路径，使得所有脚本和模块都能正常导入 ``cognitivemap`` 。

用法：在任意脚本开头添加以下两行即可：

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from cognitivemap._env import init  # noqa: E402
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # _env.py -> cognitivemap -> src -> project root


def init() -> None:
    """将项目根目录加入 ``sys.path``，使 ``script`` 包可被导入。"""
    project_root = str(_PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
