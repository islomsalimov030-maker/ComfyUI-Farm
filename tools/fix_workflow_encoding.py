"""Repair reversible CP1251/UTF-8 corruption in workflow labels and Notes only."""
import json
from pathlib import Path
import re


def repair(text):
    for _ in range(3):
        try:
            decoded = text.encode('cp1251').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if decoded == text or not re.search('[А-Яа-я]', decoded):
            break
        text = decoded
    return text


def fix(data):
    changed = 0
    if isinstance(data, list):
        return sum(fix(x) for x in data)
    if not isinstance(data, dict):
        return 0
    for key, value in data.items():
        if isinstance(value, str) and key in ('title', 'name', 'label', 'localized_name'):
            new = repair(value)
            if new != value:
                data[key] = new
                changed += 1
        elif key in ('widgets_values', 'widgets_values_named') and data.get('type') in ('Note', 'MarkdownNote'):
            if isinstance(value, list):
                for i, text in enumerate(value):
                    if isinstance(text, str):
                        new = repair(text)
                        changed += new != text
                        value[i] = new
            elif isinstance(value, dict):
                for k, text in value.items():
                    if isinstance(text, str):
                        new = repair(text)
                        changed += new != text
                        value[k] = new
        else:
            changed += fix(value)
    return changed


if __name__ == '__main__':
    total = 0
    for path in Path('workflows').rglob('*.json'):
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        count = fix(data)
        if count:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            print(f'{path}: {count}')
        total += count
    print(f'Repaired strings: {total}')
