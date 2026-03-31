// Jenkinsfile.ecr-simple
pipeline {
    agent any
    
    environment {
        AWS_ACCOUNT_ID = '693149914819'   // ❗ FIXED (no hyphens)
        AWS_REGION = 'ap-south-1'
        IMAGE_NAME = 'mask-detector'
        ECR_REPO = "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE_NAME}"
        DEPLOY_SERVER_IP = '13.233.124.42'
        DOCKER_PATH = 'C:/Program Files/Docker/Docker/resources/bin'   // ✅ Added
    }
    
   stages {

    stage('Debug Environment') {
        steps {
            bat '''
                echo PATH = %PATH%
                docker --version
            '''
        }
    }

    stage('Checkout') {
        steps {
            git url: 'https://github.com/Bharadwaj-8/Face_detection.git', branch: 'feature-automation2'
        }
    }

    stage('Build Docker Image') {
        steps {
            bat '''
                docker build -t %IMAGE_NAME%:latest .
                docker tag %IMAGE_NAME%:latest %ECR_REPO%:latest
            '''
        }
    }

   stage('Login to ECR') {
    steps {
        withCredentials([usernamePassword(
            credentialsId: 'aws-credentials',
            usernameVariable: 'AWS_ACCESS_KEY_ID',
            passwordVariable: 'AWS_SECRET_ACCESS_KEY'
        )]) {
            bat '''
                set AWS_ACCESS_KEY_ID=%AWS_ACCESS_KEY_ID%
                set AWS_SECRET_ACCESS_KEY=%AWS_SECRET_ACCESS_KEY%
                set AWS_DEFAULT_REGION=ap-south-1

                "C:\\Program Files\\Amazon\\AWSCLIV2\\aws.exe" ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 693149914819.dkr.ecr.ap-south-1.amazonaws.com
            '''
        }
    }
}
    stage('Push to ECR') {
        steps {
            bat '''
                docker push %ECR_REPO%:latest
            '''
        }
    }

   stage('Deploy to EC2') {
    steps {
        bat '''
        "C:\\Windows\\System32\\OpenSSH\\ssh.exe" ec2-user@13.233.124.42 "aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 693149914819.dkr.ecr.ap-south-1.amazonaws.com && docker pull 693149914819.dkr.ecr.ap-south-1.amazonaws.com/mask-detector:latest && docker stop mask-detector || true && docker rm mask-detector || true && docker run -d --name mask-detector -p 8081:8081 693149914819.dkr.ecr.ap-south-1.amazonaws.com/mask-detector:latest"
        '''
    }
}
}
}
