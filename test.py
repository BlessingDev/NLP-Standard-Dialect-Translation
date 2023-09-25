from jamo import h2j, j2hcj # 자모화
from cython_module import cjamo

text = "막"
jamo_str = j2hcj(h2j(text))
print(jamo_str)

result = cjamo.jamo_to_hangeul(*jamo_str)
print(result)