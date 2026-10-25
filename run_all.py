from pathlib import Path
import shutil

def main():
    root=Path(__file__).resolve().parent
    state=root/'state'
    if state.exists(): shutil.rmtree(state)
    state.mkdir(exist_ok=True)
    import subprocess,sys
    subprocess.run([sys.executable,str(root/'generate_data.py')],cwd=root,check=True)
    import part_a
    part_a.run()
    subprocess.run([sys.executable,str(root/'validate.py')],cwd=root,check=True)
    import part_b, part_c, part_d
    part_b.run(); part_c.run(); part_d.run()
if __name__=='__main__': main()
