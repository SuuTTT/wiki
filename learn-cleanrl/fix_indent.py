import glob
for file in glob.glob("/workspace/wiki/learn-cleanrl/*tutorial.py"):
    with open(file, "r") as f:
        lines = f.readlines()
    
    new_lines = []
    in_episode_block = False
    for line in lines:
        if 'if "_episode" in infos:' in line:
            indent = line.split('if')[0]
            new_lines.append(line)
            in_episode_block = True
            continue
        
        if in_episode_block:
            if line.strip() == "":
                new_lines.append(line)
                continue
            if not line.startswith(indent + " "):
                in_episode_block = False
            else:
                pass
                
    # Eaiser way to fix the replace:
