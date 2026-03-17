import traceback
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

try:
    from sam_service import get_sam_service
    print('Calling get_sam_service')
    get_sam_service()
    print('Success')
except Exception as e:
    with open('crash.log', 'w') as f:
        traceback.print_exc(file=f)
    print('Crash logged')
