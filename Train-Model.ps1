<#
.SYNOPSIS
    Train a ProjectAI model on your own text. Uses the GPU (CUDA) if available, else CPU.
.EXAMPLE
    ./Train-Model.ps1 mytext.txt -Size small -Name mymodel
.EXAMPLE
    ./Train-Model.ps1 corpus.txt -Size medium -Steps 1000
#>
param(
    [Parameter(Position = 0)][string]$File,
    [string]$Name,
    [ValidateSet('tiny', 'small', 'medium', 'large')][string]$Size = 'small',
    [int]$Steps = 300,
    [int]$Batch = 16,
    [int]$SeqLen = 128,
    [double]$Lr = 0.0003,
    [ValidateSet('auto', 'cpu', 'cuda')][string]$Device = 'auto',
    [string]$Prompt
)

$trainer = Join-Path $PSScriptRoot 'ProjectAI.Trainer'
$cliArgs = @()
if ($File) { $cliArgs += $File }
if ($Name) { $cliArgs += @('--name', $Name) }
$cliArgs += @('--size', $Size, '--steps', $Steps, '--batch', $Batch, '--seqlen', $SeqLen, '--lr', $Lr, '--device', $Device)
if ($Prompt) { $cliArgs += @('--prompt', $Prompt) }

dotnet run -c Release --project $trainer -- @cliArgs
