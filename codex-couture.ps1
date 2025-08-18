# codex_couture.ps1 ‚Äî Slay Mode Project Codex Builder (with -DebugMode)

param(
  [string]$Root    = 'Z:\dev',
  [string]$Project = 'Audio-Denoising',
  [switch]$DebugMode
)

# --- paths ---
$src     = Join-Path $Root $Project
$dst     = Join-Path $Root ($Project + '_codex')
$treeOut = Join-Path $Root  'audio_denoising_codex_tree.txt'
$logOut  = Join-Path $Root  'robocopy_audio_denoising_codex.log'
$readme  = Join-Path $dst   'CODEX_README.md'

# --- debug helper ---
function Write-D([string]$msg){
  if ($DebugMode) { Write-Host "[DEBUG] $msg" -ForegroundColor DarkCyan }
}

Write-D "Root...........: $Root"
Write-D "Project........: $Project"
Write-D "Source.........: $src"
Write-D "Destination....: $dst"
Write-D "Tree Out.......: $treeOut"
Write-D "Robo Log.......: $logOut"
Write-D "README.........: $readme"

# --- exclusions ---
$xd = @(
  "$src\__pycache__",
  "$src\.pytest_cache",
  "$src\.venv",
  "$src\.git",
  "$src\.mypy_cache",
  "$src\.ipynb_checkpoints",
  "$src\build",
  "$src\dist",
  "$src\.ruff_cache",
  "$src\.tox",
  "$src\*.egg-info"
)
$xf = @(
  '*.pyc','*.pyo','*.pyd','*.log','*.tmp','*.bak',
  'Thumbs.db','.DS_Store','*.egg-info'
)

Write-D "Exclude Dirs...: $($xd -join '; ')"
Write-D "Exclude Files..: $($xf -join '; ')"

# --- prepare destination ---
if (-not $DebugMode) {
  if (Test-Path $dst) {
    Write-Host "Cleaning destination..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $dst
  }
  New-Item -ItemType Directory -Path $dst | Out-Null
} else {
  Write-D "DebugMode ON: Skipping destination cleanup & creation until robocopy verifies structure."
  if (-not (Test-Path $dst)) {
    Write-D "Destination does not exist yet. That's fine; robocopy /L will still list what would copy."
  }
}

# --- robocopy switches ---
$roboCommon = @("/E","/R:1","/W:1","/FFT","/NFL","/NDL","/NJH","/NJS","/NP","/TEE","/LOG:$logOut")
$roboDebug  = if ($DebugMode) { @("/L") } else { @() }

Write-D "Starting copy (robocopy $(if ($DebugMode) { '/L dry-run' } else { 'live' }))..."
& robocopy "$src" "$dst" * @roboCommon @roboDebug /XD @xd /XF @xf | Out-Null

# If not debug and destination may be new, ensure it exists
if (-not $DebugMode -and -not (Test-Path $dst)) {
  New-Item -ItemType Directory -Path $dst | Out-Null
}

# --- Pretty tree (non-colored) for files/README ---
function Get-PrettyTree {
  param([string]$Path)

  if (-not (Test-Path $Path)) { return @() }

  $sep = [IO.Path]::DirectorySeparatorChar
  $lines = @("$([IO.Path]::GetFileName($Path))$sep")

  function Walk($dirPath, $prefix) {
    $children = Get-ChildItem -Force -LiteralPath $dirPath | Sort-Object { -not $_.PSIsContainer }, Name
    for ($i=0; $i -lt $children.Count; $i++) {
      $last = ($i -eq $children.Count - 1)
      $branch = if ($last) { "‚îî‚îÄ‚îÄ " } else { "‚îú‚îÄ‚îÄ " }
      $nextPrefix = if ($last) { "$prefix    " } else { "$prefix‚îÇ   " }

      if ($children[$i].PSIsContainer) {
        $lines += "$prefix$branch$($children[$i].Name)$sep"
        Walk $children[$i].FullName $nextPrefix
      } else {
        $lines += "$prefix$branch$($children[$i].Name)"
      }
    }
  }

  Walk $Path ""
  return $lines
}

# --- Colorized console tree ---
function Get-PrettyTreeColored {
  param([string]$Path)

  if (-not (Test-Path $Path)) {
    Write-D "PrettyTree: destination not found (likely debug dry run)."
    return
  }

  $sep = [IO.Path]::DirectorySeparatorChar
  Write-Host ("$([IO.Path]::GetFileName($Path))$sep") -ForegroundColor Cyan
  function Walk($dirPath, $prefix) {
    $children = Get-ChildItem -Force -LiteralPath $dirPath | Sort-Object { -not $_.PSIsContainer }, Name
    for ($i=0; $i -lt $children.Count; $i++) {
      $last = ($i -eq $children.Count - 1)
      $branch = if ($last) { "‚îî‚îÄ‚îÄ " } else { "‚îú‚îÄ‚îÄ " }
      $nextPrefix = if ($last) { "$prefix    " } else { "$prefix‚îÇ   " }

      if ($children[$i].PSIsContainer) {
        $name = "$($children[$i].Name)$sep"
        Write-Host "$prefix$branch" -NoNewline
        Write-Host $name -ForegroundColor Cyan
        Walk $children[$i].FullName $nextPrefix
      } else {
        $name = $children[$i].Name
        $ext = [IO.Path]::GetExtension($name).ToLower()
        $color = if ($ext -eq '.py') { 'Magenta' } elseif ($ext -eq '.md') { 'Green' } else { 'Yellow' }
        Write-Host "$prefix$branch" -NoNewline
        Write-Host $name -ForegroundColor $color
      }
    }
  }
  Walk $Path ""
}

# --- Write pretty tree to file (or a debug placeholder) ---
if (Test-Path $dst) {
  $prettyTree = Get-PrettyTree -Path $dst
  $prettyTree | Set-Content -Encoding UTF8 $treeOut
} else {
  @('DEBUG MODE: Dry-run only ‚Äî no destination created yet. See log:',
    $logOut) | Set-Content -Encoding UTF8 $treeOut
}

# --- LOC Summary ---
$pyFiles = if (Test-Path $dst) { Get-ChildItem -Path $dst -Recurse -Filter *.py -File } else { @() }
$totalLOC = 0
$locSummary = foreach ($f in $pyFiles) {
  $lines = (Get-Content $f | Measure-Object -Line).Lines
  $totalLOC += $lines
  [PSCustomObject]@{ File = $f.FullName.Substring($dst.Length+1); LOC = $lines }
}

Write-D "Python files..: $($pyFiles.Count)"
Write-D "Total LOC.....: $totalLOC"

# --- README content ---
$treeBlock = if (Test-Path $dst) {
  @('```txt') + (Get-PrettyTree -Path $dst) + @('```')
} else {
  @('```txt','(debug dry-run ‚Äî no destination created)','```')
}

$modeLabel = if ($DebugMode) { 'DEBUG (dry-run)' } else { 'NORMAL' }
$today = (Get-Date).ToString('yyyy-MM-dd')

$readmeBody = @()
$readmeBody += "# $Project ‚Äî CODEX"
$readmeBody += ""
$readmeBody += "> Auto-generated codex for LLMs & humans. Mode: **$modeLabel**. Updated: **$today**"
$readmeBody += ""
$readmeBody += "## Structure"
$readmeBody += $treeBlock
$readmeBody += ""
$readmeBody += "## Lines of Code Summary"
if ($locSummary -and $locSummary.Count -gt 0) {
  $readmeBody += ($locSummary | Sort-Object File | ForEach-Object { "- $($_.File): $($_.LOC) LOC" })
  $readmeBody += ""
  $readmeBody += "**Total LOC:** $totalLOC"
} else {
  $readmeBody += "- *(No Python files found ‚Äî likely debug dry-run or empty copy.)*"
}
$readmeBody -join "`r`n" | Set-Content -Encoding UTF8 $readme

# --- Console output ---
if (Test-Path $dst) {
  Get-PrettyTreeColored -Path $dst
} else {
  Write-Host "`n(No destination tree to show ‚Äî debug dry-run.)" -ForegroundColor DarkGray
}

Write-Host "`nÌ≤ñ Codex staged at: $dst"
Write-Host "Ì≥ú Pretty tree file: $treeOut"
Write-Host "Ì∑íÔ∏è  README:          $readme"
Write-Host "Ì∑æ Robocopy log:     $logOut"
if ($pyFiles) { Write-Host ("`nTotal Python LOC: " + $totalLOC) -ForegroundColor White -BackgroundColor DarkMagenta }
