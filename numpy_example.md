### q_learning 예제에 필요한 numpy 문법

```python
import numpy as np
q_table = np.zeros((2,2,2))

q_table[0,0,0] = 1
q_table[0,0,1] = 2
q_table[0,1,0] = 3
q_table[0,1,1] = 4
q_table[1,0,0] = 5
q_table[1,0,1] = 6
q_table[1,1,0] = 7
q_table[1,1,1] = 8

state = (1,0)
print(q_table[state])
print(q_table[state[0],state[1]])

# 해당 state에서 가장 큰 q값을 출력
max_Q = np.max(q_table[state])
print(max_Q)

# 해당 state에서 가장 큰 q값을 가진 action을 선택
action = np.argmax(q_table[state])
print(action)
```
