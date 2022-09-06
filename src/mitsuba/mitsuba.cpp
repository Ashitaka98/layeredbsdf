/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/core/platform.h>

// Mitsuba's "Assert" macro conflicts with Xerces' XSerializeEngine::Assert(...).
// This becomes a problem when using a PCH which contains mitsuba/core/logger.h
#if defined(Assert)
#undef Assert
#endif
#include <xercesc/parsers/SAXParser.hpp>
#include <mitsuba/core/sched_remote.h>
#include <mitsuba/core/sstream.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/appender.h>
#include <mitsuba/core/sshstream.h>
#include <mitsuba/core/shvector.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/render/renderjob.h>
#include <mitsuba/render/scenehandler.h>
#include <fstream>
#include <stdexcept>
#include <boost/algorithm/string.hpp>

#if defined(__WINDOWS__)
#include <mitsuba/core/getopt.h>
#include <winsock2.h>
#else
#include <signal.h>
#endif

#include "misc/utils.h"
#include <cnpy.h>

using XERCES_CPP_NAMESPACE::SAXParser;

using namespace mitsuba;

#define SAMPLE_RATE 25
#define SAMPLES (SAMPLE_RATE * SAMPLE_RATE * SAMPLE_RATE * SAMPLE_RATE)
#define SAMPLE_PER_QUERY 128

#define MATERIAL_TABLE_PATH "/home/lzr/Projects/layeredbsdf_mini/pyscript/material_names_table.txt"

#define OUTPUT_DIR "/home/lzr/layeredBsdfData/conductorWhwd_cpp"

ref<RenderQueue> renderQueue = NULL;

#if !defined(__WINDOWS__)
/* Handle the hang-up signal and write a partially rendered image to disk */
void signalHandler(int signal)
{
    if (signal == SIGHUP && renderQueue.get())
    {
        renderQueue->flush();
    }
    else if (signal == SIGFPE)
    {
        SLog(EWarn, "Caught a floating-point exception!");

#if defined(MTS_DEBUG_FP)
        /* Generate a core dump! */
        abort();
#endif
    }
}
#endif

class FlushThread : public Thread
{
public:
    FlushThread(int timeout) : Thread("flush"),
                               m_flag(new WaitFlag()),
                               m_timeout(timeout) {}

    void run()
    {
        while (!m_flag->get())
        {
            m_flag->wait(m_timeout * 1000);
            renderQueue->flush();
        }
    }

    void quit()
    {
        m_flag->set(true);
        join();
    }

private:
    ref<WaitFlag> m_flag;
    int m_timeout;
};

std::vector<std::string> read_material_table(std::string path)
{
    std::vector<std::string> ret;
    std::ifstream is(path.c_str());
    assert(is.is_open());
    while (!is.eof())
    {
        std::string name;
        is >> name;
        if (name != "")
            ret.push_back(name);
    }
    assert(ret.size() == 71);
    return move(ret);
}

static inline float pow2(float x)
{
    return x * x;
}

static inline float warpUniform(float x1, float x2, float x)
{
    float len = x2 - x1;
    return len * x + x1;
}

int mitsuba_app(int argc, char **argv)
{
    if (argc != 2)
    {
        cout << "usage: mitsuba ${material count}\n";
        return 1;
    }
    int count = atoi(argv[1]);
    if (count < 0)
    {
        cout << "${material count} must be an integer greater than zero\n";
        return 1;
    }
#if !defined(__WINDOWS__)
    /* Initialize signal handlers */
    struct sigaction sa;
    sa.sa_handler = signalHandler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    if (sigaction(SIGHUP, &sa, NULL))
        SLog(EError, "Could not install a custom signal handler!");
    if (sigaction(SIGFPE, &sa, NULL))
        SLog(EError, "Could not install a custom signal handler!");
#endif

    auto material_table = read_material_table(MATERIAL_TABLE_PATH);
    std::string baseDir = OUTPUT_DIR;
    if (!boost::filesystem::exists(baseDir))
    {
        boost::filesystem::create_directories(baseDir);
    }

    auto pmgr = PluginManager::getInstance();

    auto samplerProps = Properties("independent");
    samplerProps.setInteger("sampleCount", 128);
    ref<Sampler> sampler(static_cast<Sampler *>(pmgr->createObject(samplerProps)));
    sampler->configure();

    auto t = time(nullptr);
    SLog(EInfo, "random seed: %i", t);
    ref<Random> rd = new Random(t);

    float sigmaT_range[] = {0.f, 1.f, 2.f, 5.f};

    for (int i = 0; i < count; i++)
    {
        SLog(EInfo, "round %i start", i);
        float sigmaT = sigmaT_range[rd->nextUInt(sizeof(sigmaT_range) / sizeof(float))];
        float albedo[] = {1 - pow2(rd->nextFloat()),
                          1 - pow2(rd->nextFloat()),
                          1 - pow2(rd->nextFloat())};
        float g = 0;
        float alpha_0 = pow(10, warpUniform(-3.f, -0.5f, rd->nextFloat()));
        float alpha_1 = pow(10, warpUniform(-3.f, 0.f, rd->nextFloat()));
        float theta_0 = 0;
        float phi_0 = 0;
        float theta_1 = 0;
        float phi_1 = 0;

        float eta_0 = warpUniform(1.05f, 2.f, rd->nextFloat());
        std::string material_1 = material_table[rd->nextUInt(material_table.size())];

        SLog(EInfo, "[Material Preset] Selected %s", material_1.c_str());

        std::stringstream ss("");
        ss << alpha_0 << "_" << theta_0 << "_" << phi_0 << "_" << eta_0
           << "_" << sigmaT << "_" << albedo[0] << "_" << albedo[1] << "_"
           << albedo[2] << "_" << g << "_" << alpha_1 << "_" << theta_1 << "_" << phi_1 << "_" << material_1;

        std::string filename = ss.str();
        SLog(EInfo, "filename:%s", filename.c_str());

        auto layeredProps = Properties("multilayered");
        layeredProps.setBoolean("bidir", true);
        layeredProps.setString("pdf", "bidirStochTRT");
        layeredProps.setInteger("stochPdfDepth", 4);
        layeredProps.setInteger("pdfRepetitive", 1);
        layeredProps.setFloat("diffusePdf", 0.1f);
        layeredProps.setFloat("maxSurvivalProb", 1.0f);
        layeredProps.setInteger("nbLayers", 2);
        layeredProps.setVector("normal_0", Vector3(0.f, 0.f, 1.f));
        layeredProps.setSpectrum("sigmaT_0", Spectrum(sigmaT));
        layeredProps.setSpectrum("albedo_0", Spectrum(albedo));
        layeredProps.setVector("normal_1", Vector3(0.f, 0.f, 1.f));

        auto phaseProps0 = Properties("hg");
        phaseProps0.setFloat("g", g);
        ref<ConfigurableObject> phase0(pmgr->createObject(phaseProps0));

        auto surfaceProps0 = Properties("roughdielectric");
        surfaceProps0.setString("distribution", "ggx");
        surfaceProps0.setFloat("intIOR", eta_0);
        surfaceProps0.setFloat("extIOR", 1.0);
        surfaceProps0.setFloat("alpha", alpha_0);
        ref<ConfigurableObject> surface0(pmgr->createObject(surfaceProps0));

        auto surfaceProps1 = Properties("roughconductor");
        surfaceProps1.setString("distribution", "ggx");
        surfaceProps1.setString("material", material_1);
        surfaceProps1.setFloat("extEta", eta_0);
        surfaceProps1.setFloat("alpha", alpha_1);
        ref<ConfigurableObject> surface1(pmgr->createObject(surfaceProps1));

        ref<BSDF> layered = static_cast<BSDF *>(pmgr->createObject(layeredProps));
        layered->addChild("surface_0", surface0);
        layered->addChild("phase_0", phase0);
        layered->addChild("surface_1", surface1);

        surface0->setParent(layered);
        phase0->setParent(layered);
        surface1->setParent(layered);

        layered->configure();

        std::vector<float> dataset;
        dataset.reserve(SAMPLES * 7);
        time_t start = time(nullptr);
        for (int i1 = 0; i1 < SAMPLE_RATE; i1++)
        {
            for (int i2 = 0; i2 < SAMPLE_RATE; i2++)
            {
                for (int i3 = 0; i3 < SAMPLE_RATE; i3++)
                {
                    for (int i4 = 0; i4 < SAMPLE_RATE; i4++)
                    {
                        float theta_h = (i1 + rd->nextFloat()) * 2 * M_PI / SAMPLE_RATE;
                        float phi_h = (i2 + rd->nextFloat()) * M_PI_2 / SAMPLE_RATE;
                        float theta_d = (i3 + rd->nextFloat()) * 2 * M_PI / SAMPLE_RATE;
                        float phi_d = (i4 + rd->nextFloat()) * M_PI_2 / SAMPLE_RATE;
                        Omega_io wiwo = whwd_to_wiwo({theta_h, phi_h, theta_d, phi_d});
                        float theta_i = wiwo.theta1;
                        float phi_i = wiwo.phi1;
                        float theta_o = wiwo.theta2;
                        float phi_o = wiwo.phi2;
                        float x1, y1, z1, x2, y2, z2;
                        x1 = cos(theta_i) * sin(phi_i);
                        y1 = sin(theta_i) * sin(phi_i);
                        z1 = cos(phi_i);
                        x2 = cos(theta_o) * sin(phi_o);
                        y2 = sin(theta_o) * sin(phi_o);
                        z2 = cos(phi_o);
                        Vector3 wi, wo;

                        wi = Vector3(x1, y1, z1);
                        wo = Vector3(x2, y2, z2);
                        Intersection its;
                        its.wi = wi;
                        BSDFSamplingRecord bRec(its, sampler.get(), ETransportMode::ERadiance);
                        bRec.wi = wi;
                        bRec.wo = wo;
                        Spectrum accum = Spectrum(0.f);
                        int nan_count = 0;
                        for (int k = 0; k < SAMPLE_PER_QUERY; k++)
                        {
                            auto tmp = layered->eval(bRec, EMeasure::ESolidAngle);
                            if (std::isnan(tmp[0]) || std::isinf(tmp[0]) || std::isnan(tmp[1]) || std::isinf(tmp[1]) || std::isnan(tmp[2]) || std::isinf(tmp[2]))
                            {
                                nan_count++;
                            }
                            else
                            {
                                accum += tmp;
                            }
                        }
                        accum /= abs(wo[2]);
                        accum /= (SAMPLE_PER_QUERY - nan_count);
                        if (nan_count != 0)
                        {
                            SLog(EWarn, "Sampling theta_i: %.2f phi_i: %.2f theta_o: %.2f phi_o: %.2f | NaN occur %i times", theta_i, phi_i, theta_o, phi_o, nan_count);
                        }
                        dataset.push_back(theta_i);
                        dataset.push_back(phi_i);
                        dataset.push_back(theta_o);
                        dataset.push_back(phi_o);
                        dataset.push_back(accum[0]);
                        dataset.push_back(accum[1]);
                        dataset.push_back(accum[2]);
                        // SLog(EInfo, "%f %f %f %f %f %f %f", theta_i, phi_i, theta_o, phi_o, accum[0], accum[1], accum[2]);
                    }
                }
            }
        }
        std::string fullname = baseDir + "/" + filename + ".npy";
        // SLog(EInfo, "%s", fullname.c_str());
        cnpy::npy_save(fullname, &dataset[0], {SAMPLES, 7}, "w");
        time_t end = time(nullptr);
        SLog(EInfo, "round %i cost time %fs", i, difftime(end, start));
    }
    return 0;
}

int mts_main(int argc, char **argv)
{
    /* Initialize the core framework */
    Class::staticInitialization();
    Object::staticInitialization();
    PluginManager::staticInitialization();
    Statistics::staticInitialization();
    Thread::staticInitialization();
    Logger::staticInitialization();
    FileStream::staticInitialization();
    Spectrum::staticInitialization();
    Bitmap::staticInitialization();
    Scheduler::staticInitialization();
    SHVector::staticInitialization();
    SceneHandler::staticInitialization();

#if defined(__WINDOWS__)
    /* Initialize WINSOCK2 */
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData))
        SLog(EError, "Could not initialize WinSock2!");
    if (LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2)
        SLog(EError, "Could not find the required version of winsock.dll!");
#endif

#if defined(__LINUX__) || defined(__OSX__)
    /* Correct number parsing on some locales (e.g. ru_RU) */
    setlocale(LC_NUMERIC, "C");
#endif

    int retval = mitsuba_app(argc, argv);

    /* Shutdown the core framework */
    SceneHandler::staticShutdown();
    SHVector::staticShutdown();
    Scheduler::staticShutdown();
    Bitmap::staticShutdown();
    Spectrum::staticShutdown();
    FileStream::staticShutdown();
    Logger::staticShutdown();
    Thread::staticShutdown();
    Statistics::staticShutdown();
    PluginManager::staticShutdown();
    Object::staticShutdown();
    Class::staticShutdown();

#if defined(__WINDOWS__)
    /* Shut down WINSOCK2 */
    WSACleanup();
#endif

    return retval;
}

#if !defined(__WINDOWS__)
int main(int argc, char **argv)
{
    return mts_main(argc, argv);
}
#endif
