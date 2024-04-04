package experiments.cohesioncollision.evaluation

import ch.qos.logback.classic.Level
import experiments.cohesioncollision.{CohesionCollisionActions, CohesionCollisionRF, ExperimentInfo, NNFactory, StateInfo}
import it.unibo.alchemist.{AlchemistEnvironment, NoOutput, ShowEach}
import it.unibo.alchemist.loader.m2m.{JVMConstructor, SimulationModel}
import it.unibo.scarlib.core.model._
import org.slf4j.LoggerFactory
import it.unibo.scarlib.dsl.DSL._
import it.unibo.scarlib.core.model.Environment

import scala.concurrent.ExecutionContext.Implicits.global
import experiments.cohesioncollision.CohesionCollisionState.encoding
import it.unibo.scarlib.core.neuralnetwork.NeuralNetworkSnapshot

import java.nio.file.{Files, Paths}

object CohesionCollisionEval extends App {
  def measure(eval: => Unit): Long = {
    val start = System.currentTimeMillis()
    eval
    System.currentTimeMillis() - start
  }

  val argsMap = args.zipWithIndex.map { case (arg, i) => (i, arg) }.toMap
  val show = argsMap.get(0) match {
    case None => NoOutput
    case Some(steps) => new ShowEach(steps.toInt)
  }

  LoggerFactory.getLogger(classOf[SimulationModel]).asInstanceOf[ch.qos.logback.classic.Logger].setLevel(Level.OFF)
  LoggerFactory.getLogger(classOf[JVMConstructor]).asInstanceOf[ch.qos.logback.classic.Logger].setLevel(Level.OFF)

  def runEvaluationWith(count: Int, simulations: Int = 16): Unit = {
    (0 until simulations).foreach { case seed =>

      println(s"------ Simulation ${seed} with ${count} agents --------")

      implicit val configuration: Environment => Unit = (e: Environment) => {
        val env = e.asInstanceOf[AlchemistEnvironment]
        env.setOutputStrategy(show)
        env.setRandomSeed(Some(seed))
        env.setEnvironmentDefinition(s"./src/main/scala/experiments/cohesioncollision/evaluation/CohesionAndCollisionEval-${count}.yaml")
      }

      val where = s"./networks/network"

      val system = CTDELearningSystem {
        rewardFunction { new CohesionCollisionRF() }
        actionSpace { CohesionCollisionActions.toSeq() }
        dataset { ReplayBuffer[State, Action](10000) }
        agents { count }
        learningConfiguration { LearningConfiguration(dqnFactory = new NNFactory, snapshotPath = where) }
        environment { "it.unibo.alchemist.AlchemistEnvironment" }
      }

      system
        .runTest(ExperimentInfo.episodeLength, NeuralNetworkSnapshot(where, StateInfo.encoding * StateInfo.neighborhood, ExperimentInfo.hiddenSize))
    }
  }
  val simulations = List(50, 98, 162, 200, 242, 288, 338, 392, 968)
  val measuredEvaluationTime = simulations.to(LazyList).map { case count => measure { runEvaluationWith(count) } }.tapEach(time => println("Measured time: " + time))
  // create a csv from the measured times
  val csv = simulations.zip(measuredEvaluationTime).map { case (count, time) => s"${count},${time}" }.mkString("\n")
  println("Final result:")
  println(csv)
  // add a first line with the headers
  val csvHeaders = "agents,time\n" + csv
  // store the csv in a file in data/performance.csv in one line
  Files.write(Paths.get("data/performance.csv"), csvHeaders.getBytes())
  System.exit(0)
}
