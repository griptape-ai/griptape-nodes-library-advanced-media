import base64

from griptape.artifacts.audio_url_artifact import AudioUrlArtifact


def dict_to_audio_url_artifact(audio_dict: dict, audio_format: str | None = None) -> AudioUrlArtifact:
    """Convert a dictionary representation of audio to an AudioUrlArtifact."""
    from griptape_nodes.files.project_file import ProjectFileDestination

    value = audio_dict["value"]

    # If it already is an AudioUrlArtifact, just wrap and return
    if audio_dict.get("type") == "AudioUrlArtifact":
        return AudioUrlArtifact(value)

    # Remove any data URL prefix
    if "base64," in value:
        value = value.split("base64,")[1]

    # Decode the base64 payload
    audio_bytes = base64.b64decode(value)

    # Infer format/extension if not explicitly provided
    if audio_format is None:
        if "type" in audio_dict and "/" in audio_dict["type"]:
            # e.g. "audio/mpeg" -> "mpeg"
            audio_format = audio_dict["type"].split("/")[1]
        else:
            audio_format = "mp3"

    # Save to project file
    dest = ProjectFileDestination.from_situation(filename=f"audio.{audio_format}", situation="save_node_output")
    saved = dest.write_bytes(audio_bytes)

    return AudioUrlArtifact(saved.location)
