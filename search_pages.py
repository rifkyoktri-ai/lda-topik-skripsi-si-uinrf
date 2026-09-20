import sys, os, re
sys.stdout.reconfigure(encoding='utf-8')

content = open('app/main.py', encoding='utf-8').read()

# Manage Labels
manage_labels = 'Manage Labels' in content or 'manage_label' in content.lower()
print('Manage Labels exists:', manage_labels)

# Info Model
info_model = 'Info Model' in content
print('Info Model exists:', info_model)

# Sidebar selectbox options
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'selectbox' in line.lower() and 'Pilih Halaman' in content[max(0,content.find(line)-200):content.find(line)+200]:
        start = max(0, i-2)
        end = min(len(lines), i+15)
        print('\nSidebar navigation selectbox:')
        for j in range(start, end):
            print(f'  L{j+1}: {lines[j][:100]}')
        break

# All elif page blocks
page_blocks = re.findall(r'elif page == "([^"]+)"', content)
if_page = re.findall(r'if page == "([^"]+)"', content)
print('\nAll pages (if/elif blocks):')
for p in if_page + page_blocks:
    print(' ', p)
