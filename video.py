import ffmpeg


class Video:

        @staticmethod
        def video_to_frames(video_path, output_folder):
            (
                ffmpeg
                .input(video_path)
                .output(f"./{output_folder}/frame_%04d.jpg")
                .run()
            )


        @staticmethod
        def frames_to_video(frames_folder, output_path, fps):
            (
                ffmpeg
                .input(f"./{frames_folder}/frame_%04d.jpg", framerate=fps)
                .output(output_path, pix_fmt='yuv420p')
                .run()
            )

        @staticmethod
        def extract_metadata(video_path=None):
            probe = ffmpeg.probe(video_path)
            metadata = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            return metadata


        @staticmethod
        def extract_metadata_frame(video_path, output_path):
            # Ouvrir la vidéo avec FFmpeg
            video = ffmpeg.input(video_path)

            # Extraire les métadonnées pour chaque frame
            stream = video.streams[0]
            metadata = ffmpeg.probe(video_path, show_frames=True)["frames"]
            for i, frame_metadata in enumerate(metadata):
                # Traiter les métadonnées ici
                # ...
                # Enregistrer le frame traité dans un fichier
                out_filename = f"{output_path}/frame_{i}.jpg"
                frame = video.filter("select", f"gte(n,{i})").output(out_filename, vframes=1)
                ffmpeg.run(frame)


if __name__ == "__main__":
    Video.video_to_frames("./video.mp4", "/video")
    Video.frames_to_video("/video", "video_tmp.mp4",24)
