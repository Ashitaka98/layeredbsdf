import mitsuba

from mitsuba.core import *
from mitsuba.render import SceneHandler, RenderQueue, RenderJob
import multiprocessing

scheduler = Scheduler.getInstance()

fileResolver = Thread.getThread().getFileResolver()
fileResolver.appendPath('./scene/figure15/')

scene = SceneHandler.loadScene(fileResolver.resolve('kettle_all.xml'))
print(scene)
# for i in range(multiprocessing.cpu_count()):
#     scheduler.registerWorker(LocalWorker(i, 'wrk%i' % i))
# scheduler.start()

# queue = RenderQueue()

# scene.setDestinationFile('renderedResult')

# job = RenderJob('myRenderJob', scene, queue)
# job.start()

# queue.waitLeft(0)
# queue.join()

# print(Statistics.getInstance().getStats())