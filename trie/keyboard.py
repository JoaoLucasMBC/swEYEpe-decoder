def create_keyboard(path: str, key_length: float = 0.12, useString: bool = False) -> dict:
    if not useString:
        with open(path, 'r') as f:
            lines = f.readlines()
    else:
        lines = path.split('\n')
    
    keyboard = {}
    for line in lines:
        key, center = line.strip().split()

        center = center.replace('(', '').replace(')', '')

        center_x, center_y = center.split(',')
        center_x, center_y = float(center_x), float(center_y)

        keyboard[key] = (
            (center_x, center_y), # Center (x, y)
            (center_x - key_length/2, center_y + key_length/2), # Top left
            (center_x + key_length/2, center_y + key_length/2), # Top right
            (center_x + key_length/2, center_y - key_length/2), # Bottom right
            (center_x - key_length/2, center_y - key_length/2)  # Bottom left
        )
    
    return keyboard


def find_key(coords: tuple, keyboard: dict) -> str:
    for key, corners in keyboard.items():
        center, top_left, top_right, bottom_right, bottom_left = corners

        if top_left[0] <= coords[0] <= top_right[0] and top_left[1] >= coords[1] >= bottom_left[1]:
            return key

    return None


def main():
    kb = create_keyboard('./data/keyboard/keyboard.txt')

    print(kb)
    print(find_key((0.07,1.07), kb))

if __name__ == '__main__':
    main()