import sys

with open('student_alertness.py', 'r') as f:
    lines = f.readlines()

vision_content = "".join(lines[54:419])
core_content = "".join(lines[420:639])

with open('vision_utils.py', 'w') as f:
    f.write('from config_loader import *\nimport numpy as np\nimport cv2\nimport mediapipe as mp\nfrom datetime import datetime\nfrom dataclasses import dataclass, field, asdict\nfrom typing import Any, Optional\nimport copy\nimport os\nfrom pathlib import Path\nimport queue\n\n')
    f.write(vision_content)

with open('core.py', 'w') as f:
    f.write('from config_loader import *\nfrom vision_utils import *\nimport numpy as np\nimport cv2\nimport time\nimport threading\nfrom pathlib import Path\nfrom dataclasses import dataclass, field, asdict\nfrom datetime import datetime\nimport csv\nimport queue\nimport os\nfrom typing import Any, Optional\ntry:\n    import pygame\nexcept:\n    pygame = None\n\n')
    f.write(core_content)

new_student_alertness = "".join(lines[:54]) + "from vision_utils import *\nfrom core import *\n" + "".join(lines[639:])

with open('student_alertness.py', 'w') as f:
    f.write(new_student_alertness)
