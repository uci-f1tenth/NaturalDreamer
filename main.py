import gymnasium as gym
import torch
import argparse
import os
from tqdm        import tqdm
from dreamer    import Dreamer
from utils      import loadConfig, seedEverything, plotMetrics, plotMetricsPNG
from envs       import getEnvProperties, GymPixelsProcessingWrapper, CleanGymWrapper
from utils      import saveLossesToCSV, ensureParentFolders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(configFile):
    config = loadConfig(configFile)
    seedEverything(config.seed)

    runName                 = f"{config.environmentName}_{config.runName}"
    checkpointToLoad        = os.path.join(config.folderNames.checkpointsFolder, f"{runName}_{config.checkpointToLoad}")
    metricsFilename         = os.path.join(config.folderNames.metricsFolder,        runName)
    plotFilename            = os.path.join(config.folderNames.plotsFolder,          runName)
    checkpointFilenameBase  = os.path.join(config.folderNames.checkpointsFolder,    runName)
    videoFilenameBase       = os.path.join(config.folderNames.videosFolder,         runName)
    ensureParentFolders(metricsFilename, plotFilename, checkpointFilenameBase, videoFilenameBase)
    
    # Create training and evaluation environments based on config.
    observationType = getattr(config.dreamer, "observationType", "image")
    if observationType == "vector":
        from envs import RustoracerWrapper
        env = RustoracerWrapper(
            yaml_path=config.mapYaml,
            max_steps=getattr(config, "maxSteps", 10000),
        )
        envEvaluation = RustoracerWrapper(
            yaml_path=config.mapYaml,
            max_steps=getattr(config, "maxSteps", 10000),
        )
    else:
        env             = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(gym.make(config.environmentName), (64, 64))))
        envEvaluation   = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(gym.make(config.environmentName, render_mode="rgb_array"), (64, 64))))
    
    observationShape, actionSize, actionLow, actionHigh = getEnvProperties(env)
    print(f"envProperties: obs {observationShape}, action size {actionSize}, actionLow {actionLow}, actionHigh {actionHigh}")

    dreamer = Dreamer(observationShape, actionSize, actionLow, actionHigh, device, config.dreamer)
    if config.resume:
        dreamer.loadCheckpoint(checkpointToLoad)

    dreamer.environmentInteraction(env, config.episodesBeforeStart, seed=config.seed)

    iterationsNum = config.gradientSteps // config.replayRatio
    progressBar = tqdm(total=config.gradientSteps, desc="Training", unit="step")
    latestScore = None
    for _ in range(iterationsNum):
        for _ in range(config.replayRatio):
            sampledData                         = dreamer.buffer.sample(dreamer.config.batchSize, dreamer.config.batchLength)
            initialStates, worldModelMetrics    = dreamer.worldModelTraining(sampledData)
            behaviorMetrics                     = dreamer.behaviorTraining(initialStates)
            dreamer.totalGradientSteps += 1
            progressBar.update(1)

            if dreamer.totalGradientSteps % config.checkpointInterval == 0 and config.saveCheckpoints:
                suffix = f"{dreamer.totalGradientSteps/1000:.0f}k"
                dreamer.saveCheckpoint(f"{checkpointFilenameBase}_{suffix}")
                evaluationScore = dreamer.environmentInteraction(envEvaluation, config.numEvaluationEpisodes, seed=config.seed, evaluation=True, saveVideo=True, filename=f"{videoFilenameBase}_{suffix}")
                progressBar.set_postfix({"eval": f"{evaluationScore:.1f}", "step": suffix})
                print(f"\nSaved Checkpoint and Video at {suffix:>6} gradient steps. Evaluation score: {evaluationScore:>8.2f}")

        latestScore = dreamer.environmentInteraction(env, config.numInteractionEpisodes, seed=config.seed)
        progressBar.set_postfix({"reward": f"{latestScore:.1f}", "step": f"{dreamer.totalGradientSteps/1000:.0f}k"})
        if config.saveMetrics:
            metricsBase = {"envSteps": dreamer.totalEnvSteps, "gradientSteps": dreamer.totalGradientSteps, "totalReward" : latestScore}
            saveLossesToCSV(metricsFilename, metricsBase | worldModelMetrics | behaviorMetrics)
            plotMetrics(f"{metricsFilename}", savePath=f"{plotFilename}", title=f"{config.environmentName}")

    progressBar.close()
    pngFiles = plotMetricsPNG(metricsFilename, savePath=config.folderNames.plotsFolder)
    print(f"\n{'='*60}")
    print(f"  Training complete! {config.gradientSteps} gradient steps finished.")
    print(f"  Last reward: {latestScore:.2f}")
    print(f"{'='*60}")
    print(f"  Charts (HTML):  {plotFilename}.html")
    print(f"  Charts (PNG):")
    for f in pngFiles:
        print(f"    - {f}")
    print(f"  Metrics:        {metricsFilename}.csv")
    print(f"  Videos:         {config.folderNames.videosFolder}/")
    print(f"  Checkpoints:    {config.folderNames.checkpointsFolder}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="car-racing-v3.yml")
    main(parser.parse_args().config)
