__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- End of Hack ---

# Now, we simply import and run your main application code.
from working_app import main

if __name__ == "__main__":
    main()
