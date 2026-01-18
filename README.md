# Audio-spatial-processor-based-on-block-frequency-domain-convolution-using-head-transfer-function
This project is based on block wise frequency domain convolution of input audio and Head Transfer Function (HRTF) database, using FFT based Overlap Add method to achieve real-time binaural spatialized audio output while ensuring low computational complexity. The system performs fixed length block processing on the input audio stream, multiplies the audio blocks with the corresponding HRTF impulse responses in the frequency domain, and overlaps and adds them in the time domain to efficiently complete real-time convolution operations for long impulse responses.

In terms of spatial positioning, the system selects several most relevant directional responses from the HRTF database based on the azimuth and elevation angles of the target sound source, and interpolates and mixes multiple sets of HRTFs through distance weighting and weight normalization strategies to ensure smooth transitions of sound images during continuous motion and avoid auditory abrupt changes caused by direction switching. Combined with a weight smoothing update mechanism, this method can maintain stable spatial auditory perception when the sound source or head posture changes rapidly.

The entire system adopts a real-time audio stream processing architecture, supports dynamic updating of sound source position parameters, and balances audio synchronization and playback stability through delay compensation and amplitude normalization control. The project can be further linked with head posture sensors (such as IMU or VR head tracking data) to achieve an adaptive spatial listening experience based on the dynamic changes of the user's head, suitable for streaming media platforms, immersive media playback, virtual reality, and interactive audio applications. This project utilizes AI for assisted programming.
## Third-Party Data
﻿
This project is based on data derived from the SADIE Database.
﻿
Original database:
SADIE Database
Copyright 2018, University of York
Licensed under the Apache License, Version 2.0
﻿
This project uses a simplified and modified version of the data.
The original project is not affiliated with or endorsed by
the University of York.
﻿
See:  https://www.apache.org/licenses/LICENSE-2.0
﻿
