"""CPU protocol fixture only; no trained model/LLM or measured optimum."""
import json
import sys
task=json.load(sys.stdin)
parameters=task['space']['parameters']
name=next(iter(parameters),None)
if name is None:action={'partial':{}}
else:
    field=parameters[name];values=field.get('values') or [field['default']]
    action={'partial':{name:values[len(task['history'])%len(values)]}}
print(json.dumps({**action,'rationale':'interface fixture; unprovided fields selected by platform constructor'}))
