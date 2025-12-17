from exercises.squat import run_squat_live, run_squat_on_video, run_squat_on_image

def main_menu():
    print("Choose mode:")
    print("1 - Live webcam (squats)")
    print("2 - Video file")
    print("3 - Single image")

    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        run_squat_live()

    elif choice == "2":
        #video_path = input("Enter video path: ").strip()
        video_path = ("C:\\Users\\Yuval\\Desktop\\Rupin\\Year 4\\Final Project\\LastYearProject_FitVisor\\Media\\SquatYarin.mov").strip()
        run_squat_on_video(video_path)

    elif choice == "3":
        image_path = input("Enter image path: ").strip()
        run_squat_on_image(image_path)

    else:
        print("Invalid choice")


# ---- Run menu ----
if __name__ == "__main__":
    main_menu()
