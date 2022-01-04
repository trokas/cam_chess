import os
import cv2
import chess
import uuid
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from itertools import permutations
from PIL import Image


info = False

def capture() -> np.array:
    # TODO: transition capture into class?
    cam = cv2.VideoCapture(0)
    sleep(0.5)
    _, img = cam.read()
    cam.release()
    if info:
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    return img


def detect_checkerboard(img) -> list[int]:
    img_inv = np.array(256 - img, dtype=np.uint8)
    ret, corners = cv2.findChessboardCorners(img_inv, (7, 7), flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
    if ret:
        if info:
            plt.imshow(img)
            plt.scatter([p[0][0] for p in corners],
                        [p[0][1] for p in corners])
            plt.axis('off')
            plt.show()

        # Corners can come in different order, let's fix that
        four_corners = [corners[0][0], corners[6][0], corners[-1][0], corners[42][-1]]
        bottom_right = np.argmin([a + b for a, b in four_corners])
        top_left = np.argmax([a + b for a, b in four_corners])
        top_right = np.argmax([c[1] if i not in [bottom_right, top_left] else 0
                            for i, c in enumerate(four_corners)])
        bottom_left = [i for i in range(4) if i not in [bottom_right, top_left, top_right]][0]
        corners = [four_corners[bottom_left], four_corners[bottom_right],
                four_corners[top_left], four_corners[top_right]]

        return corners
    else:
        raise Exception('Please reposition your board or adjust lighting')


def capture_pipeline(corners):
    filled = capture()
    fig = adjust_perspective(filled, corners)
    return chop(fig)


def adjust_perspective(img, corners) -> np.array:
    vertices = np.stack(corners).astype('float32')
    # Also adjust to player perspective
    target = np.array([[100, 700], [700, 700], [100, 100], [700, 100]]).astype('float32')
    matrix = cv2.getPerspectiveTransform(vertices, target)
    board = cv2.warpPerspective(img, matrix, (800, 800))
    if info:
        plt.subplot(121)
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(board)
        plt.axis('off')
        plt.show()
    return board


def chop(fig) -> dict[str, np.array]:
    """Chops figures into map where key is the position name"""
    return {chess.square_name(col + 8 * (7 - row)): fig[100*row:100*(row+1), 100*col:100*(col+1)]
            for col in range(8) for row in range(8)[::-1]}


def create_dir_if_needed(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_train_data(fields, labels={}) -> None:
    """Labels dict should contain empty, white or black as key and field names as values"""
    for handle, names in labels.items():
        create_dir_if_needed(f'data/{handle}')
        for n in names:
            Image.fromarray(fields[n]).save(f'data/{handle}/{n}_{uuid.uuid4().hex}.png')


def group_by_value(mapping: dict):
    v = {}
    for key, value in sorted(mapping.items()):
        v.setdefault(value, []).append(key)
    return v


def piece_to_str(piece):
    if not piece:
        return 'empty'
    if str(piece) in 'PQNRBK':
        return 'white'
    return 'black'


def is_filled(board) -> dict[str, list[str]]:
    """Returns a dict with empty, white or black as key and list of fields as value"""
    return group_by_value({chess.square_name(i): piece_to_str(board.piece_map().get(i, ''))
                           for i in range(64)})


def predict(model, fields, return_probs=False):
    prob = model(np.stack(list(fields.values()))).numpy()
    if not return_probs:
        prob = prob.argmax(axis=1)
        if info:
            plt.imshow(prob.reshape(8, 8).T[::-1])
            plt.axis('off')
            plt.show()
    return {k: v for k, v in zip(fields.keys(), prob)}


def capture_move(model, board, corners):
    # Captures current state
    fields = capture_pipeline(corners)
    # Extracts probs from the model given current player
    probs = {k: v[1 if board.turn else 2] for k, v in predict(model, fields, True).items()}
    # Checks for legal moves
    legal_moves = [(str(m)[:2], str(m)[2:]) for m in board.legal_moves]
    # Combines probs - you should move from field a to field b
    probs_combo = [probs[b] - probs[a] for a, b in legal_moves]
    # Selects most likely move
    return list(board.legal_moves)[np.argmax(probs_combo)]
