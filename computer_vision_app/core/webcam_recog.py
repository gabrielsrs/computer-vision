import cv2
from model_loader import check_models_exist, load_models
from gesture_recognition import recognize_gesture


def main():
    if not check_models_exist():
        print(
            "Erro: Um ou mais arquivos de modelo não foram encontrados (.task, .joblib)."
        )
        return

    print("--- Carregando modelos customizados ---")
    models = load_models()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nIniciando reconhecimento CUSTOMIZADO... Pressione 'q' para sair.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        processed_frame = recognize_gesture(frame, models, timestamp_ms)

        cv2.imshow("Custom Gesture Recognition", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
