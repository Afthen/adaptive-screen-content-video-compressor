param (
    [Parameter(Mandatory=$true)][string]$InputFile
)

$ScriptDir = $PSScriptRoot
$InputPath = (Resolve-Path $InputFile).Path
$ParentDirectory = Split-Path $InputPath
$FileName = [System.IO.Path]::GetFileNameWithoutExtension($InputPath)

# Sub-directory creation
$Directory = Join-Path $ParentDirectory $FileName
New-Item -Path $Directory -ItemType Directory -Force | Out-Null

$PythonScriptPath = Join-Path $ScriptDir "archiver_core.py"
$TempAudio = Join-Path $Directory "temp_audio_$FileName.wav"
$OutputMKV = Join-Path $Directory "$($FileName)_final.mkv"
$RemuxedMP4 = Join-Path $Directory "$($FileName)_remuxed.mp4"

Write-Host "--- Starting Integrated Archive Pipeline (Meet Optimized) ---" -ForegroundColor Cyan

# Prompt the user for a dynamic start time
Write-Host ""
$TimeInput = Read-Host "Enter start time (e.g., 21:25, 01:10:05, or raw seconds. Press Enter for 0:00)"
$TimeInput = $TimeInput.Trim()

# Helper function to parse human-readable time strings into raw seconds
function Convert-TimeToSeconds {
    param ([string]$TimeStr)
    if ([string]::IsNullOrWhiteSpace($TimeStr)) { return 0.0 }
    if ($TimeStr -match '^\d+(\.\d+)?$') { return [double]$TimeStr } # Raw seconds
    
    $parts = $TimeStr.Split(':')
    if ($parts.Count -eq 2) {
        # MM:SS
        return ([int]$parts[0] * 60) + [double]$parts[1]
    } elseif ($parts.Count -eq 3) {
        # HH:MM:SS
        return ([int]$parts[0] * 3600) + ([int]$parts[1] * 60) + [double]$parts[2]
    }
    return 0.0
}

$StartTimeSecs = Convert-TimeToSeconds -TimeStr $TimeInput
Write-Host "Parsed Start Time: $StartTimeSecs seconds`n" -ForegroundColor Green

# Call Python to perform silence analysis, trimming, and master MKV creation
& python "$PythonScriptPath" "$InputPath" "$OutputMKV" "$TempAudio" "$StartTimeSecs"

# Generate Shareable MP4 (Direct WAV -> AAC)
Write-Host "`nGenerating Shareable MP4..." -ForegroundColor Cyan
& ffmpeg -hide_banner -loglevel warning -stats -y -hwaccel auto `
    -i "$OutputMKV" `
    -i "$TempAudio" `
    -map 0:v -map 1:a `
    -c:v copy -c:a aac -b:a 48k -ac 1 -movflags +faststart "$RemuxedMP4"

# Cleanup
Remove-Item $TempAudio -ErrorAction SilentlyContinue 

Write-Host "`nProcess Complete!" -ForegroundColor Green

# --- System Shutdown Sequence ---
# Initiates a 60-second countdown. Run 'shutdown /a' in PowerShell or CMD to abort.
#Write-Host "Initiating system shutdown in 60 seconds..." -ForegroundColor Yellow
#Write-Host "To cancel the shutdown, open a new terminal and run: shutdown /a" -ForegroundColor Cyan

#& shutdown.exe /s /t 60 /f /c "Archive Pipeline completed. Shutting down in 60 seconds. Run 'shutdown /a' to cancel."

Pause