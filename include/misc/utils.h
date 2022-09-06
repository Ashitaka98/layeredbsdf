#pragma once
#include <cmath>
struct Omega_io
{
    float theta1, phi1, theta2, phi2;
};

struct Omega_io_xyz
{
    float xyz[6];
};

static const float pi = acos(-1);
static Omega_io wiwo_to_whwd(Omega_io wiwo)
{
    float theta_i = wiwo.theta1;
    float phi_i = wiwo.phi1;
    float theta_o = wiwo.theta2;
    float phi_o = wiwo.phi2;
    float x1 = cos(theta_i) * sin(phi_i);
    float y1 = sin(theta_i) * sin(phi_i);
    float z1 = cos(phi_i);
    float x2 = cos(theta_o) * sin(phi_o);
    float y2 = sin(theta_o) * sin(phi_o);
    float z2 = cos(phi_o);
    float hx = x1 + x2;
    float hy = y1 + y2;
    float hz = z1 + z2;
    float norm_h = sqrt(hx * hx + hy * hy + hz * hz + 0.00001f);
    hx /= norm_h;
    hy /= norm_h;
    hz /= norm_h;
    float phi_h = acos(hz);
    float theta_h = atan2(hy, hx);
    if (theta_h < 0)
        theta_h += 2 * pi - 0.00001f;

    float wi[3] = {x1, y1, z1};
    double rot_z[3][3] = {
        {cos(-theta_h), -sin(-theta_h), 0.0},
        {sin(-theta_h), cos(-theta_h), 0.0},
        {0.0, 0.0, 1.0}};
    double rot_y[3][3] = {
        {cos(-phi_h), 0.0, sin(-phi_h)},
        {0.0, 1.0, 0.0},
        {-sin(-phi_h), 0.0, cos(-phi_h)}};

    float d_tmp[3];
    for (int i = 0; i < 3; i++)
    {
        d_tmp[i] = 0;
        for (int j = 0; j < 3; j++)
            d_tmp[i] += rot_z[i][j] * wi[j];
    }
    float d[3];
    for (int i = 0; i < 3; i++)
    {
        d[i] = 0;
        for (int j = 0; j < 3; j++)
            d[i] += rot_y[i][j] * d_tmp[j];
    }
    d[2] = d[2] < -0.9999f ? -0.9999f : d[2];
    d[2] = d[2] > 0.9999f ? 0.9999f : d[2];
    float phi_d = acos(d[2]);
    float theta_d = atan2(d[1], d[0]);
    if (theta_d < 0)
        theta_d += 2 * pi - 0.00001f;

#ifdef SH_DEBUG
    assert(theta_h >= 0 && theta_h <= 2 * pi);
    assert(theta_d >= 0 && theta_d <= 2 * pi);
    assert(phi_h >= 0 && phi_h <= pi);
    assert(phi_d >= 0 && phi_d <= pi);
#endif

    Omega_io whwd = (Omega_io){theta_h, phi_h, theta_d, phi_d};
    return whwd;
}

static Omega_io wiwo_to_whwd(float x1, float y1, float z1, float x2, float y2, float z2)
{
    float hx = x1 + x2;
    float hy = y1 + y2;
    float hz = z1 + z2;
    float norm_h = sqrt(hx * hx + hy * hy + hz * hz + 0.00001f);
    hx /= norm_h;
    hy /= norm_h;
    hz /= norm_h;
    float phi_h = acos(hz);
    float theta_h = atan2(hy, hx);
    if (theta_h < 0)
        theta_h += 2 * pi - 0.00001f;

    float wi[3] = {x1, y1, z1};
    double rot_z[3][3] = {
        {cos(-theta_h), -sin(-theta_h), 0.0},
        {sin(-theta_h), cos(-theta_h), 0.0},
        {0.0, 0.0, 1.0}};
    double rot_y[3][3] = {
        {cos(-phi_h), 0.0, sin(-phi_h)},
        {0.0, 1.0, 0.0},
        {-sin(-phi_h), 0.0, cos(-phi_h)}};

    float d_tmp[3];
    for (int i = 0; i < 3; i++)
    {
        d_tmp[i] = 0;
        for (int j = 0; j < 3; j++)
            d_tmp[i] += rot_z[i][j] * wi[j];
    }
    float d[3];
    for (int i = 0; i < 3; i++)
    {
        d[i] = 0;
        for (int j = 0; j < 3; j++)
            d[i] += rot_y[i][j] * d_tmp[j];
    }
    d[2] = d[2] < -0.9999f ? -0.9999f : d[2];
    d[2] = d[2] > 0.9999f ? 0.9999f : d[2];

    float phi_d = acos(d[2]);
    float theta_d = atan2(d[1], d[0]);
    if (theta_d < 0)
        theta_d += 2 * pi - 0.00001f;

#ifdef SH_DEBUG
    assert(theta_h >= 0 && theta_h <= 2 * pi);
    assert(theta_d >= 0 && theta_d <= 2 * pi);
    assert(phi_h >= 0 && phi_h <= pi);
    assert(phi_d >= 0 && phi_d <= pi);
#endif

    Omega_io whwd = (Omega_io){theta_h, phi_h, theta_d, phi_d};
    return whwd;
}

static Omega_io_xyz wiwo_to_whwd_xyz(float x1, float y1, float z1, float x2, float y2, float z2)
{
    float hx = x1 + x2;
    float hy = y1 + y2;
    float hz = z1 + z2;
    float norm_h = sqrt(hx * hx + hy * hy + hz * hz + 0.00001f);
    hx /= norm_h;
    hy /= norm_h;
    hz /= norm_h;
    float phi_h = acos(hz);
    float theta_h = atan2(hy, hx);
    if (theta_h < 0)
        theta_h += 2 * pi - 0.00001f;

    float wi[3] = {x1, y1, z1};
    double rot_z[3][3] = {
        {cos(-theta_h), -sin(-theta_h), 0.0},
        {sin(-theta_h), cos(-theta_h), 0.0},
        {0.0, 0.0, 1.0}};
    double rot_y[3][3] = {
        {cos(-phi_h), 0.0, sin(-phi_h)},
        {0.0, 1.0, 0.0},
        {-sin(-phi_h), 0.0, cos(-phi_h)}};

    float d_tmp[3];
    for (int i = 0; i < 3; i++)
    {
        d_tmp[i] = 0;
        for (int j = 0; j < 3; j++)
            d_tmp[i] += rot_z[i][j] * wi[j];
    }
    float d[3];
    for (int i = 0; i < 3; i++)
    {
        d[i] = 0;
        for (int j = 0; j < 3; j++)
            d[i] += rot_y[i][j] * d_tmp[j];
    }
    d[2] = d[2] < -0.9999f ? -0.9999f : d[2];
    d[2] = d[2] > 0.9999f ? 0.9999f : d[2];
    Omega_io_xyz whwd;
    whwd.xyz[0] = hx;
    whwd.xyz[1] = hy;
    whwd.xyz[2] = hz;
    whwd.xyz[3] = d[0];
    whwd.xyz[4] = d[1];
    whwd.xyz[5] = d[2];
    return whwd;
}

static Omega_io whwd_to_wiwo(Omega_io whwd)
{
    float theta_h = whwd.theta1, phi_h = whwd.phi1;
    float theta_d = whwd.theta2, phi_d = whwd.phi2;
    float x1, y1, z1, x2, y2, z2;
    x1 = cos(theta_h) * sin(phi_h);
    y1 = sin(theta_h) * sin(phi_h);
    z1 = cos(phi_h);
    x2 = cos(theta_d) * sin(phi_d);
    y2 = sin(theta_d) * sin(phi_d);
    z2 = cos(phi_d);

    float h[] = {x1, y1, z1};
    float d[] = {x2, y2, z2};

    double rot_z[3][3] = {
        {cos(theta_h), -sin(theta_h), 0.0},
        {sin(theta_h), cos(theta_h), 0.0},
        {0.0, 0.0, 1.0}};
    double rot_y[3][3] = {
        {cos(phi_h), 0.0, sin(phi_h)},
        {0.0, 1.0, 0.0},
        {-sin(phi_h), 0.0, cos(phi_h)}};

    float wi_tmp[3];
    for (int i = 0; i < 3; i++)
    {
        wi_tmp[i] = 0;
        for (int j = 0; j < 3; j++)
            wi_tmp[i] += rot_y[i][j] * d[j];
    }
    float wi[3];
    for (int i = 0; i < 3; i++)
    {
        wi[i] = 0;
        for (int j = 0; j < 3; j++)
            wi[i] += rot_z[i][j] * wi_tmp[j];
    }
    float wo[3];
    float wi_dot_h = 0;
    for (int i = 0; i < 3; i++)
        wi_dot_h += wi[i] * h[i];
    for (int i = 0; i < 3; i++)
    {
        wo[i] = 2 * wi_dot_h * h[i] - wi[i];
    }

    wi[2] = wi[2] < -0.9999f ? -0.9999f : wi[2];
    wi[2] = wi[2] > 0.9999f ? 0.9999f : wi[2];

    wo[2] = wo[2] < -0.9999f ? -0.9999f : wo[2];
    wo[2] = wo[2] > 0.9999f ? 0.9999f : wo[2];

    float theta_i, phi_i, theta_o, phi_o;
    phi_i = acos(wi[2]);
    theta_i = atan2(wi[1], wi[0]);
    if (theta_i < 0)
        theta_i += 2 * pi - 0.00001f;
    phi_o = acos(wo[2]);
    theta_o = atan2(wo[1], wo[0]);
    if (theta_o < 0)
        theta_o += 2 * pi - 0.00001f;
    return {theta_i, phi_i, theta_o, phi_o};
}

static Omega_io_xyz whwd_to_wiwo_xyz(Omega_io whwd)
{
    float theta_h = whwd.theta1, phi_h = whwd.phi1;
    float theta_d = whwd.theta2, phi_d = whwd.phi2;
    float x1, y1, z1, x2, y2, z2;
    x1 = cos(theta_h) * sin(phi_h);
    y1 = sin(theta_h) * sin(phi_h);
    z1 = cos(phi_h);
    x2 = cos(theta_d) * sin(phi_d);
    y2 = sin(theta_d) * sin(phi_d);
    z2 = cos(phi_d);

    float h[] = {x1, y1, z1};
    float d[] = {x2, y2, z2};

    double rot_z[3][3] = {
        {cos(theta_h), -sin(theta_h), 0.0},
        {sin(theta_h), cos(theta_h), 0.0},
        {0.0, 0.0, 1.0}};
    double rot_y[3][3] = {
        {cos(phi_h), 0.0, sin(phi_h)},
        {0.0, 1.0, 0.0},
        {-sin(phi_h), 0.0, cos(phi_h)}};

    float wi_tmp[3];
    for (int i = 0; i < 3; i++)
    {
        wi_tmp[i] = 0;
        for (int j = 0; j < 3; j++)
            wi_tmp[i] += rot_y[i][j] * d[j];
    }
    float wi[3];
    for (int i = 0; i < 3; i++)
    {
        wi[i] = 0;
        for (int j = 0; j < 3; j++)
            wi[i] += rot_z[i][j] * wi_tmp[j];
    }
    float wo[3];
    float wi_dot_h = 0;
    for (int i = 0; i < 3; i++)
        wi_dot_h += wi[i] * h[i];
    for (int i = 0; i < 3; i++)
    {
        wo[i] = 2 * wi_dot_h * h[i] - wi[i];
    }

    wi[2] = wi[2] < -0.9999f ? -0.9999f : wi[2];
    wi[2] = wi[2] > 0.9999f ? 0.9999f : wi[2];

    wo[2] = wo[2] < -0.9999f ? -0.9999f : wo[2];
    wo[2] = wo[2] > 0.9999f ? 0.9999f : wo[2];
    return {wi[0], wi[1], wi[2], wo[0], wo[1], wo[2]};
}