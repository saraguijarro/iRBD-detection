#!/usr/bin/env python3
import re

def fix_script(filename):
    print(f"Fixing {filename}...")
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for the x create_dataset line
        if "night_group.create_dataset('x', data=night_data['x'].values)" in line:
            indent = len(line) - len(line.lstrip())
            spaces = ' ' * indent
            
            # Add new compressed code
            new_lines.append(f"{spaces}# Combine x,y,z for better compression\n")
            new_lines.append(f"{spaces}accel_data = np.column_stack([\n")
            new_lines.append(f"{spaces}    night_data['x'].values,\n")
            new_lines.append(f"{spaces}    night_data['y'].values,\n")
            new_lines.append(f"{spaces}    night_data['z'].values\n")
            new_lines.append(f"{spaces}])\n")
            new_lines.append(f"{spaces}night_group.create_dataset(\n")
            new_lines.append(f"{spaces}    'accel',\n")
            new_lines.append(f"{spaces}    data=accel_data,\n")
            new_lines.append(f"{spaces}    compression='gzip',\n")
            new_lines.append(f"{spaces}    compression_opts=4,\n")
            new_lines.append(f"{spaces}    chunks=(10000, 3),\n")
            new_lines.append(f"{spaces}    dtype='float32'\n")
            new_lines.append(f"{spaces})\n")
            
            # Skip the next 2 lines (y and z create_dataset)
            i += 3
            continue
        
        # Add compression to timestamps line
        elif "night_group.create_dataset('timestamps'," in line and "compression" not in line:
            # Keep the line but we'll add compression on next lines
            new_lines.append(line)
            # Check if data= is on this line or next
            if "data=" not in line:
                # Add compression parameters before data
                indent = len(line) - len(line.lstrip())
                spaces = ' ' * (indent + 4)
                new_lines.append(f"{spaces}compression='gzip',\n")
                new_lines.append(f"{spaces}compression_opts=4,\n")
                new_lines.append(f"{spaces}chunks=(10000,),\n")
        else:
            new_lines.append(line)
        
        i += 1
    
    with open(filename, 'w') as f:
        f.writelines(new_lines)
    
    print(f"  ✓ Fixed {filename}")

# Fix all scripts
for v in ['v0', 'v1', 'v2', 'v3']:
    fix_script(f'preprocessing_{v}.py')

print("\n✓ All scripts fixed!")
