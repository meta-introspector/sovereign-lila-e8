# Пройти по всем модулям и найти параметры с именем 'head_scales'
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
for name, param in model.named_parameters():
    if 'head_scales' in name:
        print(name, param.data.cpu().numpy())
        # Можно также построить гистограмму
        import matplotlib.pyplot as plt
        plt.hist(param.data.cpu().numpy().flatten(), bins=20)
        plt.title(name)
        plt.show()