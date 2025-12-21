from exercises.squat import (
    run_squat_live,
    run_squat_on_video,
    run_squat_on_image,
)
from exercises.pushups import (
    run_pushups_live,
    run_pushups_on_video,
    run_pushups_on_image,
)


def choose_exercise():
    print("Choose exercise:")
    print("1 - Squats")
    print("2 - Pushups")
    return input("Enter choice (1/2): ").strip()


def choose_mode():
    print("Choose mode:")
    print("1 - Live webcam")
    print("2 - Video file")
    print("3 - Single image")
    return input("Enter choice (1/2/3): ").strip()


def main_menu():
    exercise = choose_exercise()
    if exercise not in {"1", "2"}:
        print("Invalid exercise choice")
        return

    mode = choose_mode()
    if mode not in {"1", "2", "3"}:
        print("Invalid mode choice")
        return

    if exercise == "1":  # Squats
        if mode == "1":
            run_squat_live()
        elif mode == "2":
            video_path = input("Enter video path: ").strip()
            run_squat_on_video(video_path)
        else:
            image_path = input("Enter image path: ").strip()
            run_squat_on_image(image_path)

    else:  # Pushups
        if mode == "1":
            run_pushups_live()
        elif mode == "2":
            video_path = input("Enter video path: ").strip()
            run_pushups_on_video(video_path)
        else:
            image_path = input("Enter image path: ").strip()
            run_pushups_on_image(image_path)


# ---- Run menu ----
if __name__ == "__main__":
    main_menu()
